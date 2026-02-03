import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set global aesthetic style
sns.set_theme(style="white", context="talk")

def generate_bengali_heatmap():
    print(">>> 🎨 Generating Bengali-Specific Attention Heatmap...")
    output_dir = "assets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Example Bengali poetry words: "মা" (Mother), "ভালোবাসা" (Love), "ধন্যবাদ" (Thanks), "জীবন" (Life)
    words = ["মা", "ভালোবাসা", "ধন্যবাদ", "জীবন", "কবিতা", "সুন্দর"]
    
    # Simulated attention weights highlighting "মা" (Mother) and "ভালোবাসা" (Love)
    attn_weights = np.array([[0.95, 0.88, 0.12, 0.75, 0.20, 0.05]])
    
    plt.figure(figsize=(10, 2))
    sns.heatmap(attn_weights, annot=True, xticklabels=words, yticklabels=[''], 
                cmap='YlOrRd', cbar=False, annot_kws={"size": 14, "weight": "bold"})
    
    plt.title('Cross-Lingual Attention Analysis (Bengali Poetry Verse)', pad=15, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "bengali_attention_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ SUCCESS: Bengali heatmap saved in '{save_path}'")

if __name__ == "__main__":
    generate_bengali_heatmap()
