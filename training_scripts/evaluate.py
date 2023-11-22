def evaluate(data_list, adj_list, indices, labels, model_dict, label=None):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for m in model_dict:
            model_dict[m].eval()

        num_views = len(data_list)
        probs_list = []

        for i in range(num_views):
            probs_list.append(model_dict[f"G{i+1}"](data_list[i], adj_list[i]))

        combined_prob = model_dict["F"](probs_list)


        if label is not None:
            # If labels are provided, calculate and return accuracy
            #loss = torch.mean(criterion(trval_prob, torch.LongTensor(labels[trval_idx]))).data.cpu().numpy().item()
            loss = torch.mean(criterion(combined_prob, torch.LongTensor(labels[indices])))
            accuracy = accuracy_score(labels[indices], F.softmax(combined_prob, dim=1).data.cpu().numpy().argmax(1))
            
            return loss, accuracy

        # For testing, return only the probabilities
        return F.softmax(combined_prob[indices], dim=1).data.cpu().numpy(), None
