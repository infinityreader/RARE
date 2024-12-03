import json
import matplotlib.pyplot as plt

def analysis_then_plot(file_name, figure_file_name, figure_title):
    with open(file_name, "r") as f:
        data = json.load(f)
    pat_dict = {}
    for idx, item in enumerate(data):
        found = False
        if "correct_patterns" not in item.keys():
            continue
        for pattern in item["correct_patterns"]:
            if "A3->A7" in pattern:
                pattern = "A3->A7->A3"
                found = True
            if pattern in pat_dict.keys():
                pat_dict[pattern] += 1
            else:
                pat_dict[pattern] = 1
        if found:
            valid = True
            for pattern in item["correct_patterns"]:
                if "A6" in pattern:
                    valid = False
            if valid:
                print(idx)
                # import ipdb; ipdb.set_trace()
    sorted_data = dict(sorted(pat_dict.items(), key=lambda item: item[1], reverse=True))

    # Get top 10 items and group the rest as "Others"
    top_10 = dict(list(sorted_data.items())[:10])
    others_sum = sum(list(sorted_data.values())[10:])
    top_10["Others"] = others_sum

    # Plotting bar chart
    # plt.figure(figsize=(10, 6))
    # plt.bar(top_10.keys(), top_10.values(), color='skyblue')
    # plt.xlabel('Keys')
    # plt.ylabel('Values')
    # plt.title(figure_title)
    # plt.xticks(rotation=45, ha='right')
    # plt.legend(["Top 10 trajectories"], loc="upper right")
    # plt.tight_layout()
    # plt.savefig(figure_file_name)

    # Plotting a pie chart
    colors = plt.cm.tab20.colors  # Use a colormap with 20 distinct colors
    plt.figure(figsize=(10, 10))
    plt.pie(top_10.values(), labels=top_10.keys(), autopct='%1.1f%%', startangle=140, colors=colors)
    # plt.title(figure_title)
    # plt.legend(title="Keys", loc="upper right")
    plt.tight_layout()
    plt.savefig(figure_file_name)
    # print(top_10)
    # print(sorted_data)

if __name__ == "__main__":
    # analysis_then_plot("/home/htran/generation/med_preferences/raise/save/gpt-4o-mini_MedQA_ro1_chat_v10_a1_a3_a4_fm.json",
    #                    "assets/result.png", "Top 10 common trajectories that lead to the correct answer (MedQA)")
    analysis_then_plot("/home/htran/generation/med_preferences/raise/save/Meta-Llama-3.1-70B-Instruct_SIQA_ro1_chat_a1_a3_a4_fm.json",
                       "assets/result.png", "Top 10 common trajectories that lead to the correct answer (MedQA)")
    # analysis_then_plot("save/Meta-Llama-3.1-8B-Instruct_MedQA_ro4_s3_r3_textstat_t5_chat_a1_a3_a4_a6_a7_f.json", "assets/medqa_bar.png", "Top 10 common trajectories that lead to the correct answer (MedQA)")
    # analysis_then_plot("save/Meta-Llama-3.1-8B-Instruct_STG_ro4_s3_r3_wikipedia_t5_chat_a1_a3_a4_a6_a7_f.json", "assets/stg_bar.png", "Top 10 common trajectories that lead to the correct answer (StrategyQA)")





