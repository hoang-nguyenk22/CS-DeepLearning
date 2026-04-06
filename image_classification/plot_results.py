import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

def parse_logs(log_dir='logs'):
    all_files = glob.glob(os.path.join(log_dir, "*.csv"))
    # Exclude test summaries and extended logs
    all_files = [
        f for f in all_files
        if 'test_summaries' not in f
        and 'extended_50epoch' not in f
        and 'fairlr' not in f
    ]

    records = []
    for f in all_files:
        filename = os.path.basename(f).replace('.csv', '')
        parts = filename.split('_')

        try:
            seed = int(parts[-1].replace('seed', ''))
            aug = parts[-2].replace('aug', '')
            bs = int(parts[-3].replace('bs', ''))
            strategy = parts[-4]
            model = '_'.join(parts[:-4])

            df = pd.read_csv(f)
            best_val = df['val_acc'].max()

            config = f"{model}_{strategy}_aug{aug}"

            records.append({
                'config': config,
                'model': model,
                'strategy': strategy,
                'augment': aug,
                'seed': seed,
                'best_val_acc': best_val,
                'df': df
            })
        except Exception as e:
            print(f"Skipping {f}: {e}")

    return pd.DataFrame(records)


def _set_integer_xticks(ax, n_epochs, step=2):
    """Force x-axis to show integer epoch ticks."""
    ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    ax.set_xlim(0.5, n_epochs + 0.5)


def plot_curves(records_df, output_dir='report/evaluate'):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Expand nested DataFrames for seaborn
    expanded_rows = []
    for _, row in records_df.iterrows():
        df_ep = row['df']
        for _, ep_row in df_ep.iterrows():
            expanded_rows.append({
                'model': row['model'],
                'strategy': row['strategy'],
                'augment': row['augment'],
                'seed': row['seed'],
                'epoch': int(ep_row['epoch']),
                'val_acc': ep_row['val_acc'],
                'train_acc': ep_row['train_acc'],
            })

    long_df = pd.DataFrame(expanded_rows)
    n_epochs = int(long_df['epoch'].max())
    models = long_df['model'].unique()

    # Strategy comparison
    for m in models:
        subset = long_df[(long_df['model'] == m) & (long_df['augment'] == 'light')]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=subset, x='epoch', y='val_acc', hue='strategy',
                     marker='o', markersize=4, ax=ax)
        ax.set_title(f'{m}: Strategy Comparison')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy')
        ax.set_ylim(0, 1)
        _set_integer_xticks(ax, n_epochs, step=2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{m}_strategy_compare.png'), dpi=300)
        plt.close()
        print(f"Saved {m}_strategy_compare.png")

    # Augmentation comparison
    for m in models:
        subset = long_df[(long_df['model'] == m) & (long_df['strategy'] == 'finetune')]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=subset, x='epoch', y='val_acc', hue='augment',
                     marker='s', markersize=4, ax=ax)
        ax.set_title(f'{m}: Augmentation Effect')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy')
        ax.set_ylim(0, 1)
        _set_integer_xticks(ax, n_epochs, step=2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{m}_augment_compare.png'), dpi=300)
        plt.close()
        print(f"Saved {m}_augment_compare.png")

    # Architecture comparison
    finetune_df = long_df[
        (long_df['strategy'] == 'finetune') & (long_df['augment'] == 'strong')
    ]
    if not finetune_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=finetune_df, x='epoch', y='val_acc', hue='model',
                     marker='^', markersize=4, ax=ax)
        ax.set_title('Architecture Comparison')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy')
        ax.set_ylim(0, 1)
        _set_integer_xticks(ax, n_epochs, step=2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'architecture_compare.png'), dpi=300)
        plt.close()
        print("Saved architecture_compare.png")

    # Test accuracy bar chart
    test_csv = os.path.join('logs', 'test_summaries.csv')
    if os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv)
        test_df['config_name'] = test_df['experiment'].apply(
            lambda x: x.split('_seed')[0]
        )
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=test_df, x='test_acc', y='config_name',
                    color='teal', errorbar='sd', ax=ax)
        ax.set_title('Final Test Accuracy')
        ax.set_xlabel('Test Accuracy')
        ax.set_ylabel('Strategy')
        ax.set_xlim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ground_truth_test_acc.png'), dpi=300)
        plt.close()
        print("Saved: ground_truth_test_acc.png")

    print(f"\nAll charts saved to {output_dir}/")


def main():
    print("Parsing 20-epoch logs...")
    df = parse_logs()
    if df.empty:
        print("No CSV logs found in logs/ directory.")
        return
    print(f"Found {len(df)} log files. Plotting...")
    plot_curves(df)


if __name__ == "__main__":
    main()
