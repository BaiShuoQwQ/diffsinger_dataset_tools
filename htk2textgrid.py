import textgrid
import click
from pathlib import Path
 
def lab_to_textgrid(lab_file, output_file, tier_name='phones'):
    # 读取 .lab 文件
    with lab_file.open('r', encoding='utf-8') as f:
        lines = f.readlines()
 
    # 创建一个新的 TextGrid 对象
    tg = textgrid.TextGrid()
    tier = textgrid.IntervalTier(name=tier_name)
 
    # 解析 .lab 文件内容
    for line in lines:
        if line.strip():  # 忽略空行
            xmin, xmax, label = line.split()
            xmin = float(xmin) / 10000000  # 转换回秒
            xmax = float(xmax) / 10000000  # 转换回秒
            tier.add(minTime=xmin, maxTime=xmax, mark=label)
 
    tg.append(tier)
    tg.write(output_file)
 
@click.command()
@click.option('-t', '--tier', default='phones', help='Tier name to create. Default = "phones".')
@click.option('-f', '--format', default='lab', help='Input format. Default = "lab". Note that only "lab" and "txt" are supported.')
@click.option('-i','--input_folder', required=True)
@click.option('-o','--output_folder', required=True)
def htk2textgrid(tier, format, input_folder, output_folder):
    print(input_folder)
    if format not in ['lab', 'txt']:
        raise click.UsageError('Unsupported format. Please use "lab" or "txt".')

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    input_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    for input_path in input_folder.iterdir():
        if input_path.is_file() and input_path.suffix.lower() in ['.lab', '.txt']:
            output_path = output_folder / (input_path.stem + '.TextGrid')
            lab_to_textgrid(input_path, output_path, tier)
            click.echo(f"Converted {input_path} to {output_path}")

if __name__ == "__main__":
    htk2textgrid()