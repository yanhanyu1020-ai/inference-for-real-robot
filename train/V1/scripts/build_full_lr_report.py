"""Portable report for the two full-tuning learning rates and the old LoRA reference."""
import html
import json
from pathlib import Path
import sys


def build(root):
    root=Path(root)
    summary=json.loads((root/'summary.json').read_text())
    e=html.escape
    def num(value):return '—' if value is None else f'{value:.6f}'
    pieces=['''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>HeteroVLA 全量微调学习率对照</title>
<style>body{font:16px/1.65 system-ui;max-width:1280px;margin:36px auto;padding:0 24px;color:#172331}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d5dce5;padding:9px;text-align:left}th{background:#eef3f8}img{max-width:100%}code{overflow-wrap:anywhere}.scroll{overflow:auto}h2{margin-top:30px}</style></head><body>
<h1>HeteroVLA π0.5 全量微调学习率对照</h1>
<p>两张独立单卡实验：GPU0 LR=1e-4，GPU1 LR=5e-5。训练batch32，仅实际GPU OOM后允许对应学习率新建batch16实验一次，失败配置与日志保留。seed42，warmup100，cosine衰减至峰值1/10，上限2000步，nothing_saveable。全量float32参数、无LoRA，更新约33.5亿原始参数；训练步骤使用官方loss公式与优化器，并扩展逐维指标和有限值检查。</p>
<p>训练task_01、02、04、05，验证06，排除03；沿用训练集归一化和训练增强。固定完整710验证窗口、batch16、顺序与噪声/时间种子42，关闭验证增强。每25步日志、每100步验证、每250步恢复点、最佳14维验证checkpoint；500步趋势检查及原早停规则。相对旧LoRA，batch32每步样本数加倍，全量float32与LoRA冻结bf16也存在差异。</p>
<div class="scroll"><table><tr><th>方式 / W&amp;B</th><th>LR</th><th>batch</th><th>状态 / 停止步</th><th>最佳步</th><th>验证14维loss</th><th>该步训练32维loss</th><th>Go2</th><th>Piper</th><th>无人机</th><th>累计分钟</th><th>停止原因</th></tr>''']
    runs=summary['runs']
    for label,run in [('全量',r) for r in runs]+[('最佳LoRA参考',summary['lora_reference'])]:
        m=run.get('best_val_metrics',{})
        name=run['name']
        label=f'<a href="{e(run["wandb_url"])}">{e(label+" / "+name)}</a>' if run.get('wandb_url') else e(label+' / '+name)
        batch=run.get('batch_size',16 if 'lora' in name else '—')
        cells=[label,str(run.get('peak_lr','—')),str(batch),e(run['status']+' / '+str(run.get('step',0))),str(run.get('best_step','—')),num(run.get('best_val')),num(run.get('training_loss_at_best_step')),num(m.get('go2_loss')),num(m.get('piper_loss')),num(m.get('uav_loss')),f'{run.get("elapsed_seconds",0)/60:.1f}',e(run.get('stop_reason','运行中'))]
        pieces.append('<tr>'+''.join(f'<td>{c}</td>' for c in cells)+'</tr>')
    pieces.append('</table></div><p>累计时间包含编译、加载、验证与checkpoint等待；训练loss为最佳步对应的25步日志统计，不能替代验证或动作预测。</p>')
    for run in runs:
        name=run['name'];directory=root/name
        pieces.append(f'<h2>{e(name)}</h2><p>最佳checkpoint：<code>{e(run.get("best_checkpoint","尚未产生"))}</code></p>')
        links=[]
        for filename,label in [('status.json','状态'),('effective_config.json','有效配置'),('metrics.jsonl','指标'),('compiled_memory.json','编译显存'),('stage500_review.json','500步检查'),('fallback_decision.json','OOM降batch记录')]:
            if (directory/filename).exists():links.append(f'<a href="{name}/{filename}">{label}</a>')
        if (root/'logs'/f'{name}.log').exists():links.append(f'<a href="logs/{name}.log">日志</a>')
        pieces.append('<p>'+' · '.join(links)+'</p>')
        memory=directory/'compiled_memory.json'
        if memory.exists():
            value=json.loads(memory.read_text())
            pieces.append(f'<p>训练编译估算 {value["live_estimate_bytes"]/2**30:.2f} GiB，预算 {value["safe_budget_bytes"]/2**30:.2f} GiB。XLA池比例0.90（约72 GiB），不预分配；这些是本次资源配置，不是OpenPI固定需求。编译不代表已执行训练。</p>')
        comparison=root/'action_comparison'/name/'metrics.json'
        if comparison.exists():
            data=json.loads(comparison.read_text())
            pieces.append('<h3>相同710窗口的动作预测误差</h3><p>10步积分采样，每窗口10×14动作；误差使用归一化空间，物理单位逐维列出。LoRA参考复用此前700步最佳模型在相同窗口与采样条件下的预测。</p><table><tr><th>模型</th><th>分组</th><th>归一化RMSE</th><th>归一化MAE</th></tr>')
            for model,values in data['results'].items():
                for group,m in values['groups'].items():pieces.append(f'<tr><td>{e(model)}</td><td>{e(group)}</td><td>{num(m["normalized_rmse"])}</td><td>{num(m["normalized_mae"])}</td></tr>')
            pieces.append('</table><table><tr><th>动作</th><th>单位</th><th>基座RMSE</th><th>全量RMSE</th><th>LoRA RMSE</th></tr>')
            for i,action in enumerate(data['action_names']):
                cells=[e(action),e(data['units'][i])]+[num(data['results'][label]['physical_rmse_per_dim'][i]) for label in ['base','best','lora']]
                pieces.append('<tr>'+''.join(f'<td>{cell}</td>' for cell in cells)+'</tr>')
            pieces.append('</table>')
            for group in ['go2','piper','uav']:pieces.append(f'<h3>{group}</h3><img src="action_comparison/{name}/{group}_trajectory.png" alt="基座、全量与LoRA轨迹">')
            pieces.append(f'<p><a href="action_comparison/{name}/metrics.json">动作误差JSON</a> · <a href="{e(data["wandb_url"])}">动作比较W&amp;B</a></p>')
        else:pieces.append('<p>该实验的动作比较尚未完成。</p>')
    pieces.append('<p><a href="summary.json">完整汇总JSON</a> · <a href="experiment_plan.json">实验计划</a></p><p>只有一条验证轨迹，本轮用于拟合与学习率筛选，不证明泛化或实机成功。全程离线，没有连接机器人控制。</p></body></html>')
    (root/'final_report.html').write_text(''.join(pieces))


if __name__=='__main__':build(sys.argv[1])
