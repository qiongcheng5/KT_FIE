import json
import os
import copy
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from KT2.module.data_cache import initialize_data, load_data, initialize_train_uids, get_train_uids
from KT2.KT.graph_update import update_graph
from config.config import (
    graph_dir, EM_output_dir, KT_output_dir, dataset, save_intermediate_graph, burn_in_size, root_node
)
from KC_tree.io import save_graph


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


embedding_dim = 16  
learning_rate = 1e-3
blend_weight = 0.3  

# -------------------------------
# Translation / indexing
# -------------------------------
translation_mapping = {
    'Application_Module': '应用题模块',
    'Computation_Module': '计算模块',
    'Counting_Module': '计数模块',
    'Wine_Knowledge': '白酒知识',
    'Circuit_Design': '运算放大器与电路设计',
    'Education_Theory': '教育理论与实践'
}

kc_list = list(translation_mapping.keys())
num_kcs = len(kc_list)
num_difficulties = 3
num_responses = 2


kc_embeddings = nn.Embedding(num_kcs, embedding_dim).to(device)
difficulty_embeddings = nn.Embedding(num_difficulties, embedding_dim).to(device)
response_embeddings = nn.Embedding(num_responses, embedding_dim).to(device)
projection = nn.Linear(embedding_dim * 3, embedding_dim).to(device)

optimizer = torch.optim.Adam(
    list(kc_embeddings.parameters()) +
    list(difficulty_embeddings.parameters()) +
    list(response_embeddings.parameters()) +
    list(projection.parameters()), lr=learning_rate
)


def get_embedding_torch(item):
    kc_index = kc_list.index(item['kc']) if item['kc'] in kc_list else 0
    diff_index = {"easy": 0, "medium": 1, "hard": 2}[item['difficulty']]
    resp_index = int(item['response'])

    kc_emb = kc_embeddings(torch.tensor(kc_index, device=device))
    diff_emb = difficulty_embeddings(torch.tensor(diff_index, device=device))
    resp_emb = response_embeddings(torch.tensor(resp_index, device=device))

    return projection(torch.cat([kc_emb, diff_emb, resp_emb], dim=-1))

def calculate_prior(graph, kc, parameter_graph):
    """Safe calculation of KC prior probability (used when posterior1 is None)"""
    track_path = [kc]
    while True:
        if len(graph[kc].parents) == 0:
            prior_prob = parameter_graph[kc].get('gamma_root', 0.5)
            break
        kc = graph[kc].parents[0].name
        track_path.append(kc)

    while len(track_path) > 1:
        kc = track_path.pop()
        prior_prob = prior_prob + (1 - prior_prob) * parameter_graph[kc].get('gamma', 0.5)
    return prior_prob

def evaluate(true_labels, pred_prob_results):
    auc = roc_auc_score(true_labels, pred_prob_results)
    thresholds = np.linspace(0, 1, 1000)

    accuracies = [accuracy_score(true_labels, [1 if p >= t else 0 for p in pred_prob_results]) for t in thresholds]
    best_threshold = thresholds[np.argmax(accuracies)]
    best_accuracy = max(accuracies)

    f1_scores = [f1_score(true_labels, [1 if p >= t else 0 for p in pred_prob_results]) for t in thresholds]
    best_f1 = max(f1_scores)

    print(f"AUC: {auc:.4f}, Accuracy: {best_accuracy:.4f}, F1 Score: {best_f1:.4f}, Best Threshold: {best_threshold:.3f}")
    return auc, best_accuracy, best_f1, best_threshold


def main():
    # Initialize data
    initialize_train_uids()
    uids, train_size, _ = get_train_uids()
    initialize_data()
    datas, graphs = load_data()

    # EM & KT Paths
    EM_output_path = os.path.join(EM_output_dir, dataset, f'EM_results-Set{root_node}-burn-in{burn_in_size}')
    KT_output_path = os.path.join(KT_output_dir, dataset, f'KT_results-Set{root_node}-burn-in{burn_in_size}')
    os.makedirs(KT_output_path, exist_ok=True)

    merged_mapping_path = os.path.join(graph_dir, dataset, 'merged_mapping.json')
    with open(merged_mapping_path, 'r') as f:
        mapping = json.load(f)

    # Load EM parameter graph
    with open(os.path.join(EM_output_path, 'parameter_graphs', 'parameter_graph_step_final.json'), 'r') as f:
        parameter_graph = json.load(f)

    # Root node
    root_name = translation_mapping.get(root_node, root_node)
    r_diff = parameter_graph[root_name].get('r_diff', [0.6, 0.8, 0.9])  # fallback if missing

    # Tracking results
    all_true_labels, all_pred_probs = [], []
    student_ids, questions, kcs, prior_probs, phis, epsilons, r_diffs = [], [], [], [], [], [], []

    for student_idx, uid in enumerate(uids):
        print(f"Processing student {student_idx + 1}/{len(uids)}: {uid}")
        data = copy.deepcopy(datas[uid])
        data_size = train_size.get(uid, 0)
        data = data[data_size:]  # test data
        graph = graphs[uid]

        student_true_labels, student_pred_probs = [], []

        # Iterate through exercises
        for index, item in enumerate(data):
            kc = mapping.get(item['kc'], item['kc'])

            # Incremental EM graph update
            if index > 0:
                graph, parameter_graph, r_diff = update_graph(
                    graph, index, parameter_graph, r_diff, uids, uid, mapping
                )

            # EM posterior probability
            posterior1 = getattr(graph[kc], 'posterior1', None)
            if posterior1 is None:
                posterior1 = calculate_prior(graph, kc, parameter_graph)

            # Temporal attention
            if index > 0:
                past_embeddings = torch.stack([get_embedding_torch(data[i]) for i in range(index)]).to(device)
                current_embedding = get_embedding_torch(item).to(device)
                scores = torch.matmul(past_embeddings, current_embedding)
                alphas = F.softmax(scores, dim=0)
                past_responses = torch.tensor([data[i]['response'] for i in range(index)], dtype=torch.float32, device=device)
                temporal_mastery = torch.sum(alphas * past_responses).item()
                posterior1_final = (1 - blend_weight) * posterior1 + blend_weight * temporal_mastery
            else:
                posterior1_final = posterior1

            # EM parameters for prediction
            phi = r_diff[{"easy": 0, "medium": 1, "hard": 2}[item['difficulty']]]
            epsilon = parameter_graph[kc].get('epsilon', 0.25)

            pred_prob = phi * posterior1_final + epsilon * (1 - posterior1_final)

            # Append tracking
            student_ids.append(uid)
            all_true_labels.append(item['response'])
            all_pred_probs.append(pred_prob)
            student_true_labels.append(item['response'])
            student_pred_probs.append(pred_prob)
            questions.append(item['question'])
            kcs.append(kc)
            prior_probs.append(posterior1)
            phis.append(phi)
            epsilons.append(epsilon)
            r_diffs.append(r_diff)

        # Optional: save intermediate graph
        if save_intermediate_graph:
            intermediate_graph_path = os.path.join(KT_output_path, f'intermediate_graphs/student_{uid}')
            os.makedirs(intermediate_graph_path, exist_ok=True)
            save_graph(graph, intermediate_graph_path, f'graph_student_{uid}.json')
            with open(os.path.join(intermediate_graph_path, f'parameter_graph_student_{uid}.json'), 'w') as f:
                json.dump(parameter_graph, f, indent=4, ensure_ascii=False)

    # Save all KT results
    result_df = pd.DataFrame({
        'student_id': student_ids,
        'pred_prob': all_pred_probs,
        'prior_prob': prior_probs,
        'epsilon': epsilons,
        'phi': phis,
        'true_label': all_true_labels,
        'kc': kcs,
        'question': questions
    })
    result_df.to_csv(os.path.join(KT_output_path, 'KT_results.csv'), index=False)

    # Evaluate
    auc, accuracy, f1, best_threshold = evaluate(all_true_labels, all_pred_probs)

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(all_true_labels, all_pred_probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(os.path.join(KT_output_path, 'ROC_Curve.png'))
    plt.show()


if __name__ == "__main__":
    main()
