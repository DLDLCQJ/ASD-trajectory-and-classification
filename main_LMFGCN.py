if __name__ == "__main__":
    args = parse_arguments()

    # Ensure checkpoints directory exists
    os.makedirs(args.checkpoints, exist_ok=True)

    # Load and preprocess data
    labels, data_list, y = load_data(args.path, args.view_list)

    # Run training and collect scores
    scores = run_training(args, labels, data_list, y)
