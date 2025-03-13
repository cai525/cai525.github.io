---
layout: default
title: C.P.F. |C.P.F.  
---

<style>

.lang-switch {
    text-align: right;
    font-size: 16px;
    font-family: inherit;
}

.lang-switch button {
    all: unset;  /* 彻底移除所有默认样式 */
    cursor: pointer;
    font-size: inherit;
    font-weight: normal;
    font-family: inherit;
    color: inherit;
}

.lang-switch button:hover {
    text-decoration: underline;  /* 鼠标悬停时添加下划线 */
}

.lang-switch button:focus {
    outline: none; /* 取消点击时的默认外框 */
}


</style>

<script>
document.addEventListener("DOMContentLoaded", function () {
    function toggleLanguage(lang) {
        document.querySelectorAll(".lang").forEach(el => {
            el.style.display = (el.classList.contains(lang)) ? "block" : "none";
        });
        localStorage.setItem("selectedLang", lang);
    }

    const savedLang = localStorage.getItem("selectedLang") || "zh";
    toggleLanguage(savedLang);
    
    document.getElementById("btn-zh").addEventListener("click", () => toggleLanguage("zh"));
    document.getElementById("btn-en").addEventListener("click", () => toggleLanguage("en"));
});
</script>

<div class="lang-switch">
    <button id="btn-zh">CN</button> | <button id="btn-en">EN</button>
</div>

<div class="lang zh" markdown="1">
# Blog of C.P.F.

### 🎓 教育背景

#### 中国科学技术大学 (2023.9 至今)  
硕士研究生，电子工程与信息科学系  
- 研究方向：智能音频处理与计算机听觉（声音事件检测、多模态音频理解）  
- 导师：宋彦副教授（语音及语言信息处理国家工程研究中心）  
- 预计毕业时间：2026 年 6 月  

#### 大连理工大学 (2019.9 - 2023.6)  
工学学士，电子信息工程  
- GPA: 93.20 / 100  
- 院系排名: 2 / 183（前 1%）  

### 🛠 技能

- **编程能力**: 熟悉python编程,了解c++编程;具备深度学习的项目经验，熟练使用PyTorch框架；
- **开发工具**: 了解Git协作开发、分支管理和版本管理；熟悉Linux开发环境；
- **语音算法**:  
  - 语音合成: 了解VITS，VALLE，CosyVoice等常用语音合成策略; 
  - 语音识别:了解端到端语音识别的常用策略，如CTC，Transducer以及Whisper等；
  - 了解语音及音频自监督模型预训练的常用策略;
  - 了解声纹识别相关技术;
- **英语水平**: CET4 (577)，CET6 (513)  

### 📧 联系方式
- 邮箱: [cqi525@mail.ustc.edu.cn](mailto:cqi525@mail.ustc.edu.cn)  or [good_luck_cpf@163.com](mailto:good_luck_cpf@163.com)  
- github: [GitHub: cai525](https://github.com/cai525)  

</div>

<div class="lang en" markdown="1" style="display: none;">
# Blog of C.P.F.

### 🎓 Education

#### University of Science and Technology of China (Sep. 2023 - Present)  
Master’s Student, Department of Electronic Engineering and Information Science  
- Research Focus: Intelligent audio processing and computational auditory perception (SED, multimodal audio understanding)  
- Advisor: Assoc. Prof. Yan Song (National Engineering Research Center for Speech and Language Information Processing)  
- Expected Graduation: June 2026  

#### Dalian University of Technology (Sep. 2019 - June 2023)  
Bachelor of Engineering, Electronic Information Engineering  
- GPA: 93.20 / 100  
- Rank: 2 / 183 (Top 1%)  

### 🛠 Skills

- **Programming Skills**: Proficient in Python programming, familiar with C++ programming; experienced in deep learning projects, and skilled in using the PyTorch framework.

- **Development Tools**: Familiar with Git collaborative development, branch management, and version control; proficient in the Linux development environment.

- **Speech Algorithm Stack**:
  - **Speech Synthesis**: Familiar with common speech synthesis strategies such as VITS, VALLE, and CosyVoice.
  - **Speech Recognition**: Familiar with common end-to-end speech recognition strategies, including CTC, Transducer, and Whisper.
  - **Self-Supervised Pretraining**: Understanding of common self-supervised pretraining strategies for speech and audio models.
  - **Speaker Recognition**: Familiar with related speaker recognition technologies.

- **English Proficiency**: CET4 (577), CET6 (513)  

### 📧 Contact
- e-mail: [cqi525@mail.ustc.edu.cn](mailto:cqi525@mail.ustc.edu.cn)  or [good_luck_cpf@163.com](mailto:good_luck_cpf@163.com)  
- github: [GitHub: cai525](https://github.com/cai525)  
</div>
