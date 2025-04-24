#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质-小分子复合物可视化程序
根据指定文件夹中的文件自动生成与模板样式一致的PDF报告
支持多种命名格式的文件，包括数字命名和字母命名
"""

import os
import sys
import argparse
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from pymol import cmd, stored
import pymol
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import matplotlib
from PIL import Image
import matplotlib.font_manager as fm

# 设置字体以正确显示中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置PyMOL为无界面模式
pymol.finish_launching(['pymol', '-qc'])


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="蛋白质-小分子复合物可视化")
    parser.add_argument("--input_dir", required=True, help="输入文件夹路径，包含所有需要处理的文件")
    parser.add_argument("--output", default="visualization_report.pdf", help="输出PDF文件路径")
    parser.add_argument("--font", default=None, help="指定用于中文显示的字体名称")
    parser.add_argument("--top_n", type=int, default=2, help="展示排名前N的分子，默认为2")
    
    return parser.parse_args()


def find_files(input_dir):
    """
    在输入目录中查找所有相关文件并按组织整理
    返回格式：
        {
            'protein': 'protein.pdb',
            'pockets': ['pocket_a.pdb', 'pocket_b.pdb', ...],
            'ligands': ['molecule_a.sdf', 'molecule_b.sdf', ...],
            'complexes': ['molecule_a_complex.pdbqt', 'molecule_b_complex.pdbqt', ...],
            'info_files': ['molecule_a.txt', 'molecule_b.txt', ...],
            'image_files': ['molecule_1.png', 'molecule_2.png', ...]
        }
    """
    # 查找所有相关文件
    protein_files = glob.glob(os.path.join(input_dir, "*.pdb"))
    pocket_files = glob.glob(os.path.join(input_dir, "*pocket*.pdb"))
    # 如果找不到包含pocket的文件，尝试其他可能的命名方式
    if not pocket_files:
        pocket_files = glob.glob(os.path.join(input_dir, "*Pocket*.pdb"))
    if not pocket_files:
        pocket_files = glob.glob(os.path.join(input_dir, "*site*.pdb"))
        
    ligand_files = glob.glob(os.path.join(input_dir, "*.sdf"))
    complex_files = glob.glob(os.path.join(input_dir, "*complex*.pdbqt"))
    # 如果找不到包含complex的文件，尝试查找所有pdbqt文件
    if not complex_files:
        complex_files = glob.glob(os.path.join(input_dir, "*.pdbqt"))
        
    # 查找所有txt文件，不仅限于molecule_前缀
    info_files = glob.glob(os.path.join(input_dir, "*.txt"))
    molecule_info_files = glob.glob(os.path.join(input_dir, "molecule_*.txt"))
    if molecule_info_files:
        info_files = molecule_info_files
    
    image_files = []
    
    # 查找image文件夹中的所有png文件
    image_dir = os.path.join(input_dir, "image")
    if os.path.exists(image_dir):
        image_files = glob.glob(os.path.join(image_dir, "*.png"))
    
    # 如果image文件夹不存在或为空，检查主目录中的png文件
    if not image_files:
        image_files = glob.glob(os.path.join(input_dir, "*.png"))
    
    # 从protein_files中移除pocket_files
    protein_files = [f for f in protein_files if f not in pocket_files]
    
    # 如果有多个蛋白质文件，默认使用第一个
    protein_file = protein_files[0] if protein_files else None
    
    # 组织文件
    file_groups = {
        'protein': protein_file,
        'pockets': pocket_files,
        'ligands': ligand_files,
        'complexes': complex_files,
        'info_files': info_files,
        'image_files': image_files
    }
    
    return file_groups


def extract_molecule_identifiers(file_list):
    """
    从文件列表中提取分子标识符，同时支持字母形式(a, b)和数字形式(1, 2, 3)
    返回格式: [(identifier, file_path), ...]，已排序
    """
    results = []
    
    for file_path in file_list:
        basename = os.path.basename(file_path)
        # 查找形如molecule_X的模式，其中X可以是字母或数字
        match = re.search(r'molecule_([a-zA-Z0-9]+)', basename)
        if match:
            identifier = match.group(1)
            # 如果是单个字母，转换为对应数字便于排序 (a->1, b->2, ...)
            if len(identifier) == 1 and identifier.isalpha():
                numeric_value = ord(identifier.lower()) - ord('a') + 1
                results.append((numeric_value, identifier, file_path))
            # 如果是数字，直接转换为整数
            elif identifier.isdigit():
                results.append((int(identifier), identifier, file_path))
            else:
                # 其他标识符放在最后
                results.append((9999, identifier, file_path))
    
    # 按数值排序
    results.sort()
    return [(ident, path) for _, ident, path in results]


def generate_protein_pocket_image(protein_file, pocket_file, output_file, view_params=None):
    """生成蛋白质与口袋的PyMOL可视化图像"""
    cmd.reinitialize()
    
    # 加载蛋白质和口袋
    cmd.load(protein_file, "protein")
    cmd.load(pocket_file, "pocket")
    
    # 设置显示样式
    cmd.hide("everything", "all")
    cmd.show("cartoon", "protein")  # 使用cartoon显示蛋白质二级结构
    cmd.show("surface", "pocket")
    
    # 设置颜色
    cmd.color("skyblue", "protein")  # 米色显示蛋白质
    cmd.color("orange", "pocket")    # 蓝色显示口袋
    
    # 调整透明度
    cmd.set("transparency", 0.5, "pocket")
    
    # 设置渲染质量
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_flat_sheets", 0)
    cmd.set("ray_trace_mode", 1)
    
    # 设置视角
    if view_params:
        cmd.set_view(view_params)
    else:
        cmd.orient()
        cmd.zoom("all", 2)
    
    # 设置光照和渲染选项
    cmd.set("ray_opaque_background", 0)  # 透明背景
    cmd.set("ray_shadows", 0)
    cmd.set("ray_trace_color", "black")  # 黑色背景
    
    # 生成图像
    cmd.ray(800, 600)
    cmd.png(output_file, dpi=300)
    
    return output_file


def generate_docking_image(protein_file, pocket_file, complex_file, output_file, view_params=None):
    """生成对接结果的PyMOL可视化图像"""
    cmd.reinitialize()
    
    # 加载蛋白质和复合物
    cmd.load(protein_file, "protein")
    cmd.load(complex_file, "complex")
    
    # 提取小分子部分 - 修正的部分
    # 在PDBQT文件中，小分子通常是非蛋白质残基，可以用更精确的选择
    cmd.select("ligand", "complex and not polymer")  # 选择非高分子部分作为配体
    
    # 如果上面的选择无效，尝试常见的小分子识别方法
    cmd.select("check_ligand", "ligand")
    stored.ligand_count = 0
    cmd.iterate("check_ligand", "stored.ligand_count += 1")
    
    if stored.ligand_count == 0:
        # 尝试使用残基名称选择
        cmd.select("ligand", "complex and (resn LIG or resn UNL or resn UNK or resn MOL or hetatm)")
    
    # 再次检查选择是否有效
    cmd.select("check_ligand", "ligand")
    stored.ligand_count = 0
    cmd.iterate("check_ligand", "stored.ligand_count += 1")
    
    if stored.ligand_count == 0:
        # 如果仍然为0，可能需要使用更通用的方法
        print("警告：无法准确识别小分子，将显示整个复合物")
        cmd.select("ligand", "complex")  # 最后的选择方案
    
    # 设置显示样式
    cmd.hide("everything", "all")
    cmd.show("cartoon", "protein")     # 蛋白质使用cartoon表示
    cmd.show("sticks", "ligand")       # 小分子使用棍状模型
    
    # 统一蛋白质颜色为柔和的浅灰色，进一步降低其存在感
    cmd.color("gray80", "protein")
    
    # 使用鲜艳的颜色突出显示小分子
    # 尝试按元素着色，但使用更鲜艳的颜色方案
    cmd.util.cbag("ligand")     # 使用绿色碳原子的元素着色方案
    
    # 修改碳原子颜色为更鲜艳的绿色
    cmd.color("green", "ligand and elem C")
    
    # 增加小分子的棍子半径和球体大小，使其更加突出
    cmd.set("stick_radius", 0.35, "ligand")   # 增加棍子半径
    cmd.set("stick_ball", 1, "ligand")        # 启用原子球表示
    cmd.set("stick_ball_ratio", 2.0, "ligand")  # 增加球的大小比例
    
    # 显示配体周围的关键残基，使用不同颜色
    cmd.select("binding_site", f"protein within 4.5 of ligand")
    cmd.show("sticks", "binding_site")
    cmd.set("stick_radius", 0.2, "binding_site")  # 细的结合位点棍子
    cmd.color("cyan", "binding_site")    # 使用青色显示结合位点
    
    # 设置高质量渲染
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_flat_sheets", 0)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_trace_gain", 1.5)     # 增加光线追踪增益，使颜色更鲜艳
    cmd.set("specular", 1.0)           # 增加高光效果
    cmd.set("spec_power", 80)          # 调整高光强度
    cmd.set("spec_reflect", 2.0)       # 增加反射效果
    
    # 设置视角 - 更紧密地聚焦在小分子上
    if view_params:
        cmd.set_view(view_params)
    else:
        cmd.orient("ligand")  # 以小分子为中心
        cmd.zoom("ligand", 15)  # 更近距离聚焦在小分子上
        
        # 确保可见区域大小合适，避免裁剪
        cmd.zoom("ligand around 5")  # 缩小视野，更加聚焦在小分子及其直接环境
    
    # 设置光照和渲染选项
    cmd.set("ray_opaque_background", 0)  # 透明背景
    cmd.set("ray_shadows", 1)           # 启用阴影，增强立体感
    cmd.set("ray_trace_color", "black")  # 黑色背景
    cmd.set("ambient", 0.2)             # 调整环境光，使整体效果更平衡
    cmd.set("direct", 0.8)              # 增加直接光照，强调细节
    
    # 添加表面表示以突出口袋，使用透明度较高的设置
    cmd.create("pocket_env", f"protein within 8 of ligand")
    cmd.show("surface", "pocket_env")
    cmd.set("transparency", 0.85, "pocket_env")  # 更高透明度
    cmd.color("white", "pocket_env")
    
    # 输出更大尺寸的图像，增加分辨率
    # 使用兼容所有PyMOL版本的方式设置图像大小
    cmd.viewport(2000, 1600)  # 设置视口大小
    cmd.ray()  # 渲染当前视口大小
    cmd.png(output_file, dpi=400)  # 增加DPI获得更清晰的图像
    
    print(f"已生成图像: {output_file}")
    
    return output_file


def read_text_file(file_path):
    """读取文本文件内容"""
    if os.path.exists(file_path):
        try:
            # 尝试用UTF-8编码读取
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                # 如果UTF-8失败，尝试使用GBK编码(常用于中文Windows系统)
                with open(file_path, 'r', encoding='gbk') as f:
                    return f.read()
            except UnicodeDecodeError:
                try:
                    # 最后尝试使用系统默认编码
                    with open(file_path, 'r') as f:
                        return f.read()
                except Exception as e:
                    return f"无法读取文件内容: {str(e)}"
    return "文件不存在或无法读取"


def display_docking_image(fig, gs, mol_prefix, complex_file, protein_file, pocket_files, temp_dir, font_title):
    """显示对接图像（单张图片但在PDF中显示更大）"""
    # 创建子图用于显示复合物
    ax_complex = fig.add_subplot(gs)
    
    if complex_file and os.path.exists(complex_file) and pocket_files:
        # 生成复合物图像
        complex_img = os.path.join(temp_dir, f"{mol_prefix}_complex.png")
        generate_docking_image(protein_file, pocket_files[0], complex_file, complex_img)
        
        if os.path.exists(complex_img):
            # 加载图像
            complex_image = plt.imread(complex_img)
            ax_complex.imshow(complex_image)
            
            # 在图像周围添加鲜艳的绿色边框，与小分子颜色相呼应
            rect = plt.Rectangle((-0.5, -0.5), complex_image.shape[1], complex_image.shape[0], 
                              fill=False, edgecolor='#00aa00', linewidth=4)
            ax_complex.add_patch(rect)
            
            # 添加标题
            complex_name = os.path.basename(complex_file)
            ax_complex.set_title(f"{complex_name}的可视化", **font_title)
    else:
        # 创建占位符
        ax_complex.set_facecolor('#4a7eba')
        ax_complex.text(0.5, 0.5, f"{mol_prefix}_complex.pdbqt的可视化", 
                     color='white', fontsize=12, ha='center', va='center')
    
    # 关闭坐标轴
    ax_complex.axis('off')
    
    return ax_complex

def generate_ligand_only_image(complex_file, output_file, pocket_file=None, protein_file=None, view_params=None):
    """生成小分子3D结构图像，保留蛋白质但设置高透明度，同时显示靶点口袋"""
    cmd.reinitialize()
    
    # 加载复合物文件
    cmd.load(complex_file, "complex")
    
    # 如果提供了蛋白质文件和口袋文件，也加载它们
    if protein_file and os.path.exists(protein_file):
        cmd.load(protein_file, "full_protein")
    
    if pocket_file and os.path.exists(pocket_file):
        cmd.load(pocket_file, "pocket")
    
    # 提取小分子部分 - 与display_docking_image函数使用相同的选择方法
    cmd.select("ligand", "complex and not polymer")
    
    # 如果上面的选择无效，尝试常见的小分子识别方法
    cmd.select("check_ligand", "ligand")
    stored.ligand_count = 0
    cmd.iterate("check_ligand", "stored.ligand_count += 1")
    
    if stored.ligand_count == 0:
        # 尝试使用残基名称选择
        cmd.select("ligand", "complex and (resn LIG or resn UNL or resn UNK or resn MOL or hetatm)")
    
    # 再次检查选择是否有效
    cmd.select("check_ligand", "ligand")
    stored.ligand_count = 0
    cmd.iterate("check_ligand", "stored.ligand_count += 1")
    
    if stored.ligand_count == 0:
        # 如果仍然为0，可能需要使用更通用的方法
        print("警告：无法准确识别小分子，将显示整个复合物")
        cmd.select("ligand", "complex")
    
    # 选择蛋白质部分
    cmd.select("protein", "complex and polymer")
    
    # 设置显示样式 - 先隐藏所有内容
    cmd.hide("everything", "all")
    
    # 分别设置蛋白质、小分子和口袋的显示方式
    cmd.show("sticks", "ligand")     # 小分子使用棍状模型
    cmd.show("surface", "protein")   # 蛋白质使用surface表示
    
    # 如果加载了口袋，显示口袋
    if cmd.get_names("objects").count("pocket") > 0:
        cmd.show("surface", "pocket")
        cmd.color("orange", "pocket")  # 设置口袋颜色为深蓝色
        cmd.set("transparency", 0.2, "pocket")  # 设置口袋透明度
    else:
        # 如果没有加载独立的口袋文件，尝试基于小分子位置定义口袋
        cmd.select("computed_pocket", f"protein within 6 of ligand")
        cmd.show("surface", "computed_pocket")
        cmd.color("orange", "computed_pocket")
        cmd.set("transparency", 0.2, "computed_pocket")
    
    # 统一蛋白质颜色并设置透明度
    cmd.color("skyblue", "protein")  # 设置蛋白质颜色为淡黄色
    cmd.set("transparency", 0.5, "protein")  # 设置蛋白质透明度
    
    # 使用鲜艳的颜色突出显示小分子
    cmd.util.cbag("ligand")     # 使用绿色碳原子的元素着色方案
    
    # 修改碳原子颜色为更鲜艳的绿色
    cmd.color("green", "ligand and elem C")
    
    # 增加小分子的棍子半径和球体大小
    cmd.set("stick_radius", 0.45, "ligand")   # 增加棍子半径
    cmd.set("stick_ball", 1, "ligand")        # 启用原子球表示
    cmd.set("stick_ball_ratio", 2.2, "ligand")  # 增加球的大小比例
    
    # 设置高质量渲染
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_flat_sheets", 0)
    cmd.set("ray_trace_mode", 1)
    
    # 设置视角，以小分子为中心
    if view_params:
        cmd.set_view(view_params)
    else:
        cmd.orient("ligand")
        cmd.zoom("ligand", 4)  # 调整缩放比例以显示更多周围蛋白质
    
    # 设置光照和渲染选项
    cmd.set("ray_opaque_background", 0)  # 透明背景
    cmd.set("ray_shadows", 1)
    cmd.set("ray_trace_color", "black")
    cmd.set("ambient", 0.25)
    cmd.set("direct", 0.75)
    cmd.set("specular", 1.0)
    cmd.set("spec_power", 80)
    cmd.set("spec_reflect", 2.0)
    
    # 输出图像
    cmd.viewport(2000, 1600)  # 设置视口大小
    cmd.ray()
    cmd.png(output_file, dpi=300)
    
    print(f"已生成小分子3D结构图像: {output_file}")
    
    return output_file


def create_visualization_pdf(args, file_groups):
    """创建包含所有可视化内容的PDF报告"""
    # 创建临时文件夹存储中间图像
    temp_dir = "temp_visualization"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 检测系统是否有中文字体，如果没有则尝试使用其他通用字体
    # 如果用户指定了字体，则优先使用
    if args.font:
        plt.rcParams['font.sans-serif'] = [args.font] + plt.rcParams['font.sans-serif']
        print(f"使用用户指定的字体: {args.font}")
    else:
        # 尝试找出所有可能支持中文的字体
        font_list = fm.findSystemFonts()
        chinese_fonts = []
        
        for font_path in font_list:
            try:
                font = fm.FontProperties(fname=font_path)
                if font.get_name() in ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi',
                                      'PingFang SC', 'Heiti SC', 'Source Han Sans CN', 'Noto Sans CJK SC',
                                      'WenQuanYi Micro Hei']:
                    chinese_fonts.append(font.get_name())
            except:
                pass
        
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = chinese_fonts + plt.rcParams['font.sans-serif']
            print(f"检测到的中文字体: {', '.join(chinese_fonts[:3])}{'...' if len(chinese_fonts) > 3 else ''}")
        else:
            print("警告：未检测到中文字体，可能导致中文显示为方框。")
            # 尝试设置通用字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
    
    # 尝试直接加载TrueType字体
    custom_font = None
    try:
        # 检查是否有支持中文的TTF字体文件
        for font_path in [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # Linux 常见中文字体路径
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Linux Noto
            'C:/Windows/Fonts/simhei.ttf',  # Windows 中文字体
            'C:/Windows/Fonts/msyh.ttc',  # Windows 雅黑
            '/System/Library/Fonts/PingFang.ttc',  # macOS 中文字体
        ]:
            if os.path.exists(font_path):
                custom_font = fm.FontProperties(fname=font_path)
                print(f"加载中文字体文件: {font_path}")
                break
    except Exception as e:
        print(f"加载TTF字体文件失败: {e}")
    
    # 设置字体属性
    try:
        if custom_font:
            font_title = {'fontsize': 14, 'fontweight': 'bold', 'fontproperties': custom_font}
            font_suptitle = {'fontsize': 16, 'fontweight': 'bold', 'fontproperties': custom_font}
        else:
            font_title = {'fontsize': 14, 'fontweight': 'bold', 'fontfamily': plt.rcParams['font.sans-serif'][0]}
            font_suptitle = {'fontsize': 16, 'fontweight': 'bold', 'fontfamily': plt.rcParams['font.sans-serif'][0]}
    except Exception as e:
        print(f"设置字体出错: {e}")
        font_title = {'fontsize': 14, 'fontweight': 'bold'}
        font_suptitle = {'fontsize': 16, 'fontweight': 'bold'}
    
    # 获取文件
    protein_file = file_groups['protein']
    pocket_files = file_groups['pockets']
    complex_files = file_groups['complexes']
    info_files = file_groups['info_files']
    image_files = file_groups['image_files']
    
    if not protein_file:
        print("错误: 未找到蛋白质PDB文件")
        return
    
    if not pocket_files:
        print("错误: 未找到口袋PDB文件")
        return
    
    # 创建PDF报告
    with PdfPages(args.output) as pdf:
        # ===== 第一页：靶点口袋和预测分子 =====
        fig1 = plt.figure(figsize=(12, 8))
        gs1 = GridSpec(2, 1, height_ratios=[1, 1], figure=fig1)
        
        # 第一行：靶点口袋
        gs_top = gs1[0].subgridspec(1, 2)
        
        # 靶点口袋标题和图片 - 现在使用左对齐
        ax_top_left = fig1.add_subplot(gs_top[0])
        ax_top_left.text(0.0, 0.5, "靶点口袋：", fontsize=16, fontweight='bold', 
                         horizontalalignment='left', verticalalignment='center')
        ax_top_left.axis('off')
        
        ax_top_right = fig1.add_subplot(gs_top[1])
        
        # 口袋图像
        if pocket_files:
            pocket_img = os.path.join(temp_dir, "pocket_visualization.png")
            generate_protein_pocket_image(protein_file, pocket_files[0], pocket_img)
            pocket_image = plt.imread(pocket_img)
            
            # 在图像周围添加蓝色边框
            height, width = pocket_image.shape[:2]
            border_width = 20  # 边框宽度（像素）
            
            # 创建带蓝色边框的图像
            bordered_image = np.ones((height + 2*border_width, width + 2*border_width, 4))
            bordered_image[:, :, :3] = [0.3, 0.5, 0.9]  # 蓝色
            bordered_image[:, :, 3] = 1  # 完全不透明
            
            # 将原始图像放在蓝色边框中间
            bordered_image[border_width:border_width+height, border_width:border_width+width] = pocket_image
            
            ax_top_right.imshow(bordered_image)
            # 可以添加额外的说明文字如果需要
            ax_top_right.text(0.5, -0.05, "橙色为口袋区", 
                    ha='center', va='top',
                    transform=ax_top_right.transAxes,
                    **font_title)
        
        ax_top_right.axis('off')
        
        # 第二行：模型预测的分子
        gs_bottom = gs1[1].subgridspec(2, 1, height_ratios=[1, 5])
        
        # 收集并排序所有分子图像文件
        molecule_images = []
        molecule_identifiers = extract_molecule_identifiers(image_files)
        for _, image_path in molecule_identifiers:
            if os.path.basename(image_path).endswith(".png"):
                molecule_images.append(image_path)
        
        # 分子预测标题 - 动态显示找到的分子数量，现在使用左对齐
        ax_bottom_title = fig1.add_subplot(gs_bottom[0])
        molecule_count = len(molecule_images)
        title_text = f"模型预测的{molecule_count}个分子：" if molecule_count > 0 else "模型预测的分子："
        ax_bottom_title.text(0.0, 0.5, title_text, fontsize=16, fontweight='bold',
                           horizontalalignment='left', verticalalignment='center')
        ax_bottom_title.axis('off')
        
        if molecule_images:
            # 计算网格的行列数
            max_columns = min(5, len(molecule_images))  # 最多5列
            num_rows = (len(molecule_images) + max_columns - 1) // max_columns  # 向上取整
            
            # 创建多行网格
            gs_molecules = gs_bottom[1].subgridspec(num_rows, max_columns)
            
            # 展示分子图像
            for i, image_path in enumerate(molecule_images):
                row = i // max_columns
                col = i % max_columns
                ax = fig1.add_subplot(gs_molecules[row, col])
                
                if os.path.exists(image_path):
                    img = plt.imread(image_path)
                    ax.imshow(img)
                    
                    # 在图像周围添加蓝色边框
                    rect = plt.Rectangle((-0.5, -0.5), img.shape[1], img.shape[0], 
                                        fill=False, edgecolor='#4a7eba', linewidth=3)
                    ax.add_patch(rect)
                    
                    # 添加标题（文件名）
                    basename = os.path.basename(image_path)
                    ax.set_title(basename, **font_title, pad=10)
                else:
                    # 图像不存在的处理
                    ax.set_facecolor('#4a7eba')
                    ax.text(0.5, 0.5, f"图像不存在", 
                          color='white', fontsize=12, ha='center', va='center')
                
                ax.axis('off')
                
            # 处理剩余的空单元格
            for i in range(len(molecule_images), num_rows * max_columns):
                row = i // max_columns
                col = i % max_columns
                ax = fig1.add_subplot(gs_molecules[row, col])
                ax.axis('off')  # 保持空白
        else:
            # 没有分子图像的情况
            ax_no_images = fig1.add_subplot(gs_bottom[1])
            ax_no_images.text(0.5, 0.5, "未找到分子图像", 
                            fontsize=14, ha='center', va='center')
            ax_no_images.axis('off')
        
        fig1.tight_layout()
        pdf.savefig(fig1)
        plt.close(fig1)
        
        # ===== 第二页：对接得分前N的分子 =====
        # 提取并排序所有标识符
        ligand_identifiers = extract_molecule_identifiers(file_groups['ligands'])
        complex_identifiers = extract_molecule_identifiers(file_groups['complexes'])
        info_identifiers = extract_molecule_identifiers(file_groups['info_files'])
        
        # 根据实际分子数量和参数值确定top_n值
        top_n = min(args.top_n, len(ligand_identifiers))
        
        # 只取前N个分子
        top_ligand_identifiers = ligand_identifiers[:top_n]
        
        # 如果没有足够的分子，显示警告
        if len(top_ligand_identifiers) < args.top_n:
            print(f"警告: 只找到了{len(top_ligand_identifiers)}个分子，少于请求的前{args.top_n}个")
        
        # 计算高度比例：标题行和分子行
        height_ratios = [1] + [4] * len(top_ligand_identifiers)  # 增加分子行的高度比例
        
        # 修改第二页的图像大小，保持与第一页宽度一致
        fig2 = plt.figure(figsize=(12, 10 + 4*len(top_ligand_identifiers)))  # 将宽度从16改为12，与第一页保持一致
        gs2 = GridSpec(1 + len(top_ligand_identifiers), 1, height_ratios=height_ratios, figure=fig2)
        
        # 标题行 - 左对齐并动态显示实际分子数量
        ax_title = fig2.add_subplot(gs2[0])
        ax_title.text(0.0, 0.5, f"其中对接得分前{len(top_ligand_identifiers)}的分子：", 
                     fontsize=16, fontweight='bold',
                     horizontalalignment='left', verticalalignment='center')
        ax_title.axis('off')
        
        # 依次展示每个分子的信息
        for row, (identifier, ligand_file) in enumerate(top_ligand_identifiers, 1):
            # 提取分子名称和查找相关文件
            mol_prefix = f"molecule_{identifier}"
            
            # 查找对应的复合物文件
            complex_file = None
            for ident, path in complex_identifiers:
                if ident == identifier:
                    complex_file = path
                    break
            
            # 查找对应的信息文件
            info_file = None
            for ident, path in info_identifiers:
                if ident == identifier:
                    info_file = path
                    break
            
            # 查找图像文件
            img_files = [f for f in image_files if os.path.basename(f).startswith(mol_prefix) and f.endswith('.png')]
            img_file = img_files[0] if img_files else None
            
            # 为当前分子创建一个子网格
            # 使用2行布局：第一行为信息框，第二行为图像
            gs_mol = gs2[row].subgridspec(2, 1, height_ratios=[1, 4])
            
            # 分子信息文本框 (第一行)
            ax_info = fig2.add_subplot(gs_mol[0])
            
            if info_file and os.path.exists(info_file):
                # 读取文本文件
                info_text = read_text_file(info_file)
                
                # 创建橙色背景
                ax_info.set_facecolor('#f6923d')
                
                # 显示文本内容 - 框居中但文本靠左对齐
                ax_info.text(0.05, 0.5, info_text, 
                          color='black', fontsize=10, ha='left', va='center',
                          transform=ax_info.transAxes,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='#f6923d', alpha=0.8))
            else:
                # 创建橙色占位符
                ax_info.set_facecolor('#f6923d')
                ax_info.text(0.05, 0.5, f"{mol_prefix}.txt里面的各项信息展示出来", 
                          color='black', fontsize=12, ha='left', va='center')
            
            ax_info.axis('off')
            
            # 创建第2行的2列网格布局，用于放置两张图
            # 修改：添加边距控制，使图像更小并居中对齐
            gs_images = gs_mol[1].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
            
            # 小分子3D结构图 (左)
            ax_ligand_3d = fig2.add_subplot(gs_images[0])
            
            if complex_file and os.path.exists(complex_file):
                # 生成小分子3D结构图
                ligand_3d_img = os.path.join(temp_dir, f"{mol_prefix}_ligand_3d.png")
                generate_ligand_only_image(complex_file, ligand_3d_img, pocket_files[0] if pocket_files else None, protein_file)
                
                if os.path.exists(ligand_3d_img):
                    # 加载图像
                    ligand_3d_image = plt.imread(ligand_3d_img)
                    
                    # 修改：调整图像大小，使其在视图中占据90%左右
                    # 创建宽度为90%的子轴域
                    subax_ligand = ax_ligand_3d.inset_axes([0.05, 0.05, 0.9, 0.9])  # [left, bottom, width, height]
                    subax_ligand.imshow(ligand_3d_image)
                    subax_ligand.axis('off')
                    
                    # 添加紫色边框
                    rect = plt.Rectangle((-0.5, -0.5), ligand_3d_image.shape[1], ligand_3d_image.shape[0], 
                                      fill=False, edgecolor='#8800cc', linewidth=3)
                    subax_ligand.add_patch(rect)
                    
                    # 设置标题，并调整位置使其更靠近图像
                    ax_ligand_3d.set_title("小分子3D结构", **font_title, y=0.85)  # y值调小，使标题更靠近图像
                    ax_ligand_3d.axis('off')
                else:
                    # 创建占位符，并调整标题位置
                    ax_ligand_3d.set_facecolor('#8800cc')
                    ax_ligand_3d.text(0.5, 0.5, "小分子3D结构", 
                                   color='white', fontsize=12, ha='center', va='center')
                    ax_ligand_3d.set_title("", y=0.85)  # 空标题但保持位置一致性
            else:
                # 创建占位符，并调整标题位置
                ax_ligand_3d.set_facecolor('#8800cc')
                ax_ligand_3d.text(0.5, 0.5, "小分子3D结构\n(未找到复合物文件)", 
                               color='white', fontsize=12, ha='center', va='center')
                ax_ligand_3d.set_title("", y=0.85)  # 空标题但保持位置一致性
            
            ax_ligand_3d.axis('off')
            
            # 分子复合物可视化 (右)
            ax_complex = fig2.add_subplot(gs_images[1])
            
            if complex_file and os.path.exists(complex_file) and pocket_files:
                # 生成复合物图像
                complex_img = os.path.join(temp_dir, f"{mol_prefix}_complex.png")
                generate_docking_image(protein_file, pocket_files[0], complex_file, complex_img)
                
                if os.path.exists(complex_img):
                    # 加载图像
                    complex_image = plt.imread(complex_img)
                    
                    # 修改：调整图像大小，使其在视图中占据90%左右
                    # 创建宽度为90%的子轴域
                    subax_complex = ax_complex.inset_axes([0.05, 0.05, 0.9, 0.9])  # [left, bottom, width, height]
                    subax_complex.imshow(complex_image)
                    subax_complex.axis('off')
                    
                    # 在图像周围添加鲜艳的绿色边框，与小分子颜色相呼应
                    rect = plt.Rectangle((-0.5, -0.5), complex_image.shape[1], complex_image.shape[0], 
                                      fill=False, edgecolor='#00aa00', linewidth=4)
                    subax_complex.add_patch(rect)
                    
                    # 添加标题并调整位置使其更靠近图像
                    complex_name = os.path.basename(complex_file)
                    ax_complex.set_title(f"{complex_name}的可视化", **font_title, y=0.85)  # y值调小，使标题更靠近图像
            else:
                # 创建占位符，并调整标题位置
                ax_complex.set_facecolor('#4a7eba')
                ax_complex.text(0.5, 0.5, f"{mol_prefix}_complex.pdbqt的可视化", 
                             color='white', fontsize=12, ha='center', va='center')
                ax_complex.set_title("", y=0.85)  # 空标题但保持位置一致性
            
            # 关闭坐标轴
            ax_complex.axis('off')
        
        # 调整布局，增加垂直间距
        fig2.tight_layout(pad=2.5)  # 增加内边距
        
        # 保存到PDF
        pdf.savefig(fig2)
        plt.close(fig2)
    
    # 清理临时文件
    temp_files = glob.glob(os.path.join(temp_dir, "*.png"))
    for file_path in temp_files:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)
    
    print(f"可视化报告已生成: {args.output}")

def main():
    """主函数"""
    try:
        args = parse_arguments()
        
        # 检查输入文件夹是否存在
        if not os.path.exists(args.input_dir) or not os.path.isdir(args.input_dir):
            print(f"错误: 输入文件夹 '{args.input_dir}' 不存在或不是一个目录")
            return 1
        
        # 查找所有相关文件
        file_groups = find_files(args.input_dir)
        
        # 显示找到的文件信息
        print("\n==== 找到的文件 ====")
        print(f"蛋白质PDB文件: {os.path.basename(file_groups['protein']) if file_groups['protein'] else '未找到'}")
        print(f"口袋PDB文件: {', '.join([os.path.basename(f) for f in file_groups['pockets']]) if file_groups['pockets'] else '未找到'}")
        print(f"小分子SDF文件: {', '.join([os.path.basename(f) for f in file_groups['ligands']]) if file_groups['ligands'] else '未找到'}")
        print(f"复合物PDBQT文件: {', '.join([os.path.basename(f) for f in file_groups['complexes']]) if file_groups['complexes'] else '未找到'}")
        print(f"信息文本文件: {', '.join([os.path.basename(f) for f in file_groups['info_files']]) if file_groups['info_files'] else '未找到'}")
        print(f"分子图像文件: {len(file_groups['image_files'])} 个文件")
        
        # 显示使用的字体信息
        print("\n==== 字体配置 ====")
        print(f"当前sans-serif字体: {plt.rcParams['font.sans-serif'][:3]}{'...' if len(plt.rcParams['font.sans-serif']) > 3 else ''}")
        
        # 创建可视化
        create_visualization_pdf(args, file_groups)
        print(f"\n成功生成PDF报告: {args.output}")
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        print(traceback.format_exc())
        return 1
    return 0
    
def run_visualization(input_dir, output_file="visualization_report.pdf", top_n=2, font=None):

    class Args:
        def __init__(self, input_dir, output, top_n, font):
            self.input_dir = input_dir
            self.output = output
            self.top_n = top_n
            self.font = font
    
    args = Args(input_dir=input_dir, 
                output=output_file, 
                top_n=top_n, 
                font=font)
    
    file_groups = find_files(args.input_dir)
    create_visualization_pdf(args, file_groups)

if __name__ == "__main__":
    sys.exit(main())