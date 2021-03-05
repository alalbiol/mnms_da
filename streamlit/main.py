import os
import torch
import streamlit as st
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

rootdir = '../notebooks/'
coral_covs_dirs = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if "loader_covs_coral_" in subdir:
            if subdir.split("/")[-1] not in coral_covs_dirs:
                coral_covs_dirs.append(subdir.split("/")[-1])


for covs_dir in coral_covs_dirs:
    st.write(f"## {covs_dir}")

    # -- Seen
    A_train_covs = torch.load(os.path.join(rootdir, covs_dir, "A_train_covs.pt"), map_location=torch.device('cpu'))
    A_test_covs = torch.load(os.path.join(rootdir, covs_dir, "A_test_covs.pt"), map_location=torch.device('cpu'))
    B_train_covs = torch.load(os.path.join(rootdir, covs_dir, "B_train_covs.pt"), map_location=torch.device('cpu'))
    B_test_covs = torch.load(os.path.join(rootdir, covs_dir, "B_test_covs.pt"), map_location=torch.device('cpu'))
    # -- Unseen
    C_test_covs = torch.load(os.path.join(rootdir, covs_dir, "C_test_covs.pt"), map_location=torch.device('cpu'))
    D_test_covs = torch.load(os.path.join(rootdir, covs_dir, "D_test_covs.pt"), map_location=torch.device('cpu'))

    all_covs = {"A_train_covs": A_train_covs, "A_test_covs": A_test_covs, "B_train_covs": B_train_covs,
                "B_test_covs": B_test_covs, "C_test_covs": C_test_covs, "D_test_covs": D_test_covs}

    all_corals = {}


    def coral_with_covs(source_c, target_c, num_classes=4):
        loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))
        loss = loss / (4 * num_classes * num_classes)
        return loss


    progress_bar = st.progress(0)
    for indx, (case1, case2) in enumerate(list(combinations(all_covs, 2))):
        all_corals[f"{case1}_vs_{case2}"] = []
        for case1_cov in all_covs[case1]:
            for case2_cov in all_covs[case2]:
                coral_loss = coral_with_covs(case1_cov, case2_cov)
                all_corals[f"{case1}_vs_{case2}"].append(coral_loss.item())
        progress_bar.progress(min(indx / len(list(combinations(all_covs, 2))), 1.0))
    progress_bar.empty()

    st.write("Min Coral Value: ", str(np.array(all_corals["A_train_covs_vs_B_train_covs"]).min()))
    st.write("Max Coral Value: ", str(np.array(all_corals["A_train_covs_vs_B_train_covs"]).max()))

    df_AteBte = pd.DataFrame({'coral': all_corals["A_test_covs_vs_B_test_covs"],
                              'info': ["AteBte"] * len(all_corals["A_test_covs_vs_B_test_covs"])})
    df_AtrBtr = pd.DataFrame({'coral': all_corals["A_train_covs_vs_B_train_covs"],
                              'info': ["AtrBtr"] * len(all_corals["A_train_covs_vs_B_train_covs"])})
    df_AtrAte = pd.DataFrame({'coral': all_corals["A_train_covs_vs_A_test_covs"],
                              'info': ["AtrAte"] * len(all_corals["A_train_covs_vs_A_test_covs"])})
    df_BtrBte = pd.DataFrame({'coral': all_corals["B_train_covs_vs_B_test_covs"],
                              'info': ["BtrBte"] * len(all_corals["B_train_covs_vs_B_test_covs"])})

    df_1_coral = pd.concat([df_AteBte, df_AtrBtr, df_AtrAte, df_BtrBte])

    fig, ax = plt.subplots()
    sns.kdeplot(
        data=df_1_coral, x="coral", hue="info",
        fill=False, common_norm=False,  # palette="crest",
        alpha=.5, linewidth=1, ax=ax
    )
    ax.set_title("Coral Loss Comparison using Vendor A and B as reference")
    st.pyplot(fig)

    df_AtrBte = pd.DataFrame({'coral': all_corals["A_train_covs_vs_B_test_covs"],
                              'info': ["AtrBte"] * len(all_corals["A_train_covs_vs_B_test_covs"])})
    df_AtrCte = pd.DataFrame({'coral': all_corals["A_train_covs_vs_C_test_covs"],
                              'info': ["AtrCte"] * len(all_corals["A_train_covs_vs_C_test_covs"])})
    df_AtrDte = pd.DataFrame({'coral': all_corals["A_train_covs_vs_D_test_covs"],
                              'info': ["AtrDte"] * len(all_corals["A_train_covs_vs_D_test_covs"])})
    df_AtrAte = pd.DataFrame({'coral': all_corals["A_train_covs_vs_A_test_covs"],
                              'info': ["AtrAte"] * len(all_corals["A_train_covs_vs_A_test_covs"])})

    df_1_coral = pd.concat([df_AtrBte, df_AtrCte, df_AtrDte, df_AtrAte])

    fig, ax = plt.subplots()
    sns.kdeplot(
        data=df_1_coral, x="coral", hue="info",
        fill=False, common_norm=False,  # palette="crest",
        alpha=.5, linewidth=1, ax=ax
    )
    ax.set_title("Coral Loss Comparison using Vendor A training as reference")
    st.pyplot(fig)

    df_BtrBte = pd.DataFrame({'coral': all_corals["B_train_covs_vs_B_test_covs"],
                              'info': ["BtrBte"] * len(all_corals["B_train_covs_vs_B_test_covs"])})
    df_BtrCte = pd.DataFrame({'coral': all_corals["B_train_covs_vs_C_test_covs"],
                              'info': ["BtrCte"] * len(all_corals["B_train_covs_vs_C_test_covs"])})
    df_BtrDte = pd.DataFrame({'coral': all_corals["B_train_covs_vs_D_test_covs"],
                              'info': ["BtrDte"] * len(all_corals["B_train_covs_vs_D_test_covs"])})
    df_BtrAte = pd.DataFrame({'coral': all_corals["A_test_covs_vs_B_train_covs"],
                              'info': ["BtrAte"] * len(all_corals["A_test_covs_vs_B_train_covs"])})

    df_1_coral = pd.concat([df_BtrBte, df_BtrCte, df_BtrDte, df_BtrAte])

    fig, ax = plt.subplots()
    sns.kdeplot(
        data=df_1_coral, x="coral", hue="info",
        fill=False, common_norm=False,  # palette="crest",
        alpha=.5, linewidth=1, ax=ax
    )
    ax.set_title("Coral Loss Comparison using Vendor B training as reference")
    st.pyplot(fig)

    df_AtrBtr = pd.DataFrame({'coral': all_corals["A_train_covs_vs_B_train_covs"],
                              'info': ["AtrBtr"] * len(all_corals["A_train_covs_vs_B_train_covs"])})
    df_1_coral = pd.concat([df_AtrBtr])

    fig, ax = plt.subplots()
    sns.kdeplot(
        data=df_1_coral, x="coral", hue="info",
        fill=False, common_norm=False,  # palette="crest",
        alpha=.5, linewidth=1, ax=ax
    )
    ax.set_title("Coral Loss Comparison between A train and B train")
    st.pyplot(fig)
