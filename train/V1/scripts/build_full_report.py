"""Create a portable report for successful or blocked full-tuning experiments."""
import html
import json
from pathlib import Path
import sys


def build(root):
    root = Path(root)
    summary = json.loads((root/'summary.json').read_text())
    full, lora = summary['runs'][0], summary['lora_reference']
    e = html.escape
    def number(value):
        return '尚无结果' if value is None else f'{value:.6f}'
    pieces = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>HeteroVLA 全量微调对照报告</title>',
        '<style>body{font:16px/1.65 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#172331}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d5dce5;padding:9px;text-align:left}th{background:#eef3f8}img{max-width:100%}code{overflow-wrap:anywhere}h2{margin-top:32px}</style>',
        '<h1>HeteroVLA π0.5 全量微调与LoRA对照</h1>',
        '<p>峰值LR=1e-4，batch16，seed42，warmup100步，cosine衰减至1e-5，上限2000步；nothing_saveable。训练task_01、02、04、05，验证task_06，排除task_03。复用训练集归一化和相同710窗口验证缓存，固定验证噪声、时间种子与样本顺序，关闭验证增强。</p>',
        '<p>全量模型使用原始Gemma架构，无LoRA适配器，更新视觉、语言、动作专家、投影和时间MLP全部约33.5亿参数。沿用官方全量模式的float32参数；LoRA冻结参数使用bf16，报告保留这一精度差异。每25步记录指标，每100步完整验证，每250步保存恢复点并保留最佳真实14维验证checkpoint；500步检查训练趋势，沿用5次无≥1%改善和过拟合停止规则。</p>',
        '<table><tr><th>方式 / W&amp;B</th><th>状态</th><th>最佳步</th><th>验证14维loss</th><th>该步训练32维loss</th><th>Go2</th><th>Piper</th><th>无人机</th><th>累计分钟</th><th>停止步 / 原因</th></tr>']
    for label, run in [('全量', full), ('最佳LoRA', lora)]:
        metrics = run.get('best_val_metrics', {})
        link = f'<a href="{e(run["wandb_url"])}">{label}</a>' if run.get('wandb_url') else label
        cells = [link, e(run['status']), str(run.get('best_step', '—')), number(run.get('best_val')),
            number(run.get('training_loss_at_best_step')), number(metrics.get('go2_loss')),
            number(metrics.get('piper_loss')), number(metrics.get('uav_loss')),
            f'{run.get("elapsed_seconds",0)/60:.1f}', e(str(run.get('step', 0))+' / '+run.get('stop_reason','—'))]
        pieces.append('<tr>'+''.join(f'<td>{cell}</td>' for cell in cells)+'</tr>')
    pieces += ['</table><p>训练loss为最佳checkpoint对应步数的官方32维日志值（25步统计）；累计时间包含脚本计入的编译、加载和验证，不是纯训练时间。</p>', '<h2>Checkpoint</h2>']
    for label, run in [('全量',full),('LoRA',lora)]:
        pieces.append(f'<p>{label}: <code>{e(run.get("best_checkpoint","尚未产生checkpoint"))}</code></p>')
    directory = root/full['name']
    memory = directory/'compiled_memory.json'
    if memory.exists():
        data = json.loads(memory.read_text())
        pieces.append(f'<h2>启动资源记录</h2><p>编译训练步骤估算显存 {data["live_estimate_bytes"]/2**30:.2f} GiB；本次共享GPU安全预算 {data["safe_budget_bytes"]/2**30:.2f} GiB。分配池比例0.68、54.4 GiB与启动56 GiB门槛是本次资源限制，不是OpenPI固有要求。若资源检查失败则不加载权重、不执行参数更新；配置和日志保留。</p>')
    path = root/'action_comparison'/full['name']/'metrics.json'
    if path.exists():
        comparison = json.loads(path.read_text())
        pieces.append('<h2>相同710窗口的实际动作预测误差</h2><p>每窗口10步、14维动作，官方10步积分采样，seed42及批次偏移相同。分组误差在归一化动作空间中计算，物理误差按不同单位的维度分别给出。LoRA列复用上一轮相同窗口、相同采样条件的700步预测。</p><table><tr><th>模型</th><th>组</th><th>归一化RMSE</th><th>归一化MAE</th></tr>')
        for label, result in comparison['results'].items():
            for group, values in result['groups'].items():
                pieces.append(f'<tr><td>{e(label)}</td><td>{e(group)}</td><td>{number(values["normalized_rmse"])}</td><td>{number(values["normalized_mae"])}</td></tr>')
        pieces.append('</table><table><tr><th>动作</th><th>单位</th><th>基座RMSE</th><th>全量RMSE</th><th>LoRA RMSE</th></tr>')
        for i, name in enumerate(comparison['action_names']):
            cells = [e(name),e(comparison['units'][i])] + [number(comparison['results'][label]['physical_rmse_per_dim'][i]) for label in ['base','best','lora']]
            pieces.append('<tr>'+''.join(f'<td>{cell}</td>' for cell in cells)+'</tr>')
        pieces.append('</table><p>轨迹图展示每窗口首个动作；误差指标覆盖全部710×10个动作。</p>')
        for group in ['go2','piper','uav']:
            rel = f'action_comparison/{full["name"]}/{group}_trajectory.png'
            pieces.append(f'<h2>{group}轨迹</h2><img src="{e(rel)}" alt="基座、全量与LoRA动作轨迹">')
        pieces.append(f'<p><a href="{e(comparison["wandb_url"])}">动作比较W&amp;B</a> · <a href="action_comparison/{full["name"]}/metrics.json">动作误差JSON</a></p>')
    else:
        pieces.append('<p>动作比较尚未完成；若全量训练尚未更新参数，本报告不提供虚构的验证指标或预测轨迹。</p>')
    pieces += [f'<h2>记录</h2><p><a href="summary.json">summary.json</a> · <a href="{e(full["name"])}/effective_config.json">有效配置</a> · <a href="{e(full["name"])}/compiled_memory.json">编译显存</a> · <a href="{e(full["name"])}/stage500_review.json">500步检查</a></p>',
        '<p>只有一条验证轨迹，本实验用于拟合和微调方式比较，不证明泛化或实机成功。全程离线，没有连接机器人执行控制。</p></html>']
    (root/'final_report.html').write_text(''.join(pieces))


if __name__ == '__main__':
    build(sys.argv[1])
