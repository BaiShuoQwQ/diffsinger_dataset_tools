tools_config：
    singer:         当前数据的歌手名, 决定数据处理时的文件夹
    #language:      当前数据的语言, 目前仅支持'zh'即中文
    dictionary:     当前使用的字典文件
    SOFA_ckpt:      当前使用的SOFA模型
    FBL_ckpt:       当前使用的FBL模型
    SOME_ckpt:      当前使用的SOME模型
    

文件夹：
    work_audios:                复制要处理的音频于此
    data/singer/original:       原始音频
    data/singer/norm:           响度匹配音频
    data/singer/wavs:           切片音频
    data/singer/lab:            歌词lab标注
    data/singer/textgrids:      TextGrid标注
    data/singer/bad:            自动筛选较差的标注
    data/singer/ds:             ds文件
    data/singer/singer_dataset: 最终数据集

使用:
    ckpt: 将rmvpe, SOFA, FBL, SOME模型及其可能的配置文件放入对应文件夹; 下方已列出推荐模型
        rmvpe: https://github.com/yxlllc/RMVPE/releases/tag/230917
        SOFA: https://github.com/qiuqiao/SOFA/releases/tag/v1.0.1
        FBL: https://github.com/autumn-DL/FoxBreatheLabeler/releases/tag/0.2
        SOME: https://github.com/openvpi/SOME/releases/tag/v1.0.0-baseline
    tools_config.yaml: 使用前按实际修改
    quick_start.py: 自动将data/work_audios内的汉语普通话歌声wav文件制作成最终数据集