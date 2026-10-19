import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
print("脚本启动")
try:
        # 读取数据
    df = pd.read_csv("service_tickets.csv")

    # ========== 1.基础清洗 ==========
    # 标记状态空值
    df["is_null_status"] = df["resolved_status"].isna()
    # 过滤有效样本：resolved_status 0 /1，排除空值
    df_valid = df[df["resolved_status"].isin([0,1])].copy()
    df_valid["resolved_status"] = df_valid["resolved_status"].astype(int)

    total_valid = len(df_valid)
    success_cnt = (df_valid["resolved_status"] ==1).sum()
    fail_cnt = (df_valid["resolved_status"] ==0).sum()
    complain_rate = fail_cnt / total_valid
    print(f"有效工单:{total_valid},成功:{success_cnt},投诉(答错):{fail_cnt}")
    print(f"整体投诉率:{complain_rate:.2%}")
    print(f"投诉带来总损失分钟：{fail_cnt * 120} min")

    # ==========2.规则深度 rule_version_depth 分析 ==========
    # 按规则深度分组：计算投诉率、平均对话轮次
    depth_stats = df_valid.groupby("rule_version_depth").agg(
        total=("resolved_status","count"),
        complain=("resolved_status", lambda x: (x==0).sum()),
        avg_dialog_turns=("dialog_turns","mean")
    ).reset_index()
    depth_stats["complain_rate"] = depth_stats["complain"] / depth_stats["total"]

    # 划分：简易规则(0‑10) /中等(11‑20) /复杂(21‑30)
    def depth_bin(d):
        if d <=10:
            return "0‑10简易规则"
        elif d <=20:
            return "11‑20中等规则"
        else:
            return "21‑30复杂规则"
    df_valid["depth_bin"] = df_valid["rule_version_depth"].apply(depth_bin)
    bin_stats = df_valid.groupby("depth_bin").agg(
        total=("resolved_status","count"),
        complain=("resolved_status",lambda x:(x==0).sum()),
        avg_turns=("dialog_turns","mean")
    ).reset_index()
    bin_stats["complain_rate"] = bin_stats["complain"] / bin_stats["total"]
    print("\n====按规则深度分组统计====")
    print(bin_stats)

    # ==========3.各大区维度分析 ==========
    group_stats = df_valid.groupby("service_group").agg(
        total=("resolved_status","count"),
        complain=("resolved_status",lambda x:(x==0).sum()),
        avg_turns=("dialog_turns","mean")
    ).reset_index()
    group_stats["complain_rate"] = group_stats["complain"] / group_stats["total"]
    print("\n====各大区统计====")
    print(group_stats)

    # ==========4.商品类目维度 ==========
    cat_stats = df_valid.groupby("category").agg(
        total=("resolved_status","count"),
        complain=("resolved_status",lambda x:(x==0).sum()),
        avg_turns=("dialog_turns","mean")
    ).reset_index()
    cat_stats["complain_rate"] = cat_stats["complain"] / cat_stats["total"]
    cat_stats = cat_stats.sort_values("complain_rate",ascending=False)
    print("\n====类目统计(投诉率降序)====")
    print(cat_stats)

    # ==========5.时间差：规则同步与用户咨询时间差 ==========
    # 转换时间戳
    df["inquiry_time"] = pd.to_datetime(df["inquiry_time"])
    df["rule_synced_at"] = pd.to_datetime(df["rule_synced_at"])
    # 计算：用户咨询距离最近一次规则同步的小时数
    df["diff_hour"] = (df["inquiry_time"] - df["rule_synced_at"]).dt.total_seconds() / 3600
    df_valid = df_valid.merge(df[["ticket_id","diff_hour"]], on="ticket_id")

    # 按同步时差分组
    def time_bin(h):
        if h < 0:
            return "咨询早于规则同步(异常)"
        elif h <=24:
            return "同步24h内"
        else:
            return "同步超过24h"
    df_valid["time_bin"] = df_valid["diff_hour"].apply(time_bin)
    time_stats = df_valid.groupby("time_bin").agg(
        total=("resolved_status","count"),
        complain=("resolved_status",lambda x:(x==0).sum()),
        avg_turns=("dialog_turns","mean")
    ).reset_index()
    time_stats["complain_rate"] = time_stats["complain"] / time_stats["total"]
    print("\n====规则同步时差分组====")
    print(time_stats)

    # ==========6.绘图输出 ==========
    fig, axes = plt.subplots(2,2,figsize=(14,10))

    # 子图1：不同规则深度分组投诉率
    axes[0,0].bar(bin_stats["depth_bin"], bin_stats["complain_rate"])
    axes[0,0].set_title("不同规则深度分组‑投诉率")
    axes[0,0].set_ylabel("投诉率")

    #子图2：各大区投诉率
    axes[0,1].bar(group_stats["service_group"], group_stats["complain_rate"])
    axes[0,1].set_title("各大区‑投诉率")
    axes[0,1].set_ylabel("投诉率")

    #子图3：类目投诉率Top
    axes[1,0].barh(cat_stats["category"], cat_stats["complain_rate"])
    axes[1,0].set_title("商品类目投诉率")

    #子图4：规则同步时差投诉率
    axes[1,1].bar(time_stats["time_bin"], time_stats["complain_rate"])
    axes[1,1].set_title("规则同步时间差‑投诉率")

    plt.tight_layout()
    plt.savefig("analysis_result.png",dpi=150)
    plt.close()

    # ==========7.成本测算：优化方案收益模拟 ==========
    high_risk = df_valid[df_valid["depth_bin"]=="21‑30复杂规则"]
    high_risk_cnt = len(high_risk)
    high_risk_complain = (high_risk["resolved_status"]==0).sum()
    origin_loss = high_risk_complain * 120
    #模拟全部高风险工单提前转人工，不再产生投诉
    new_cost = high_risk_cnt *12
    save_min = origin_loss - new_cost
    print("\n====模拟优化收益(针对复杂规则工单提前转人工)====")
    print(f"复杂规则工单总数:{high_risk_cnt},其中投诉:{high_risk_complain}")
    print(f"原损失：{origin_loss}分钟；全部提前转人工成本：{new_cost}分钟")
    print(f"净节约时间：{save_min} 分钟；若save_min>0代表方案划算")
except Exception as e:
    traceback.print_exc()
    print("脚本执行出错！")