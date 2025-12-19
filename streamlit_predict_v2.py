import streamlit as st
import pickle
import pandas as pd

# 设置页面配置（标题、图标）
st.set_page_config(page_title="医疗费用预测", page_icon="🏥")

# 定义简介页面函数
def introduce_page():
    """当选择简介页面时，呈现该函数的内容"""
    st.write("# 欢迎使用医疗费用预测应用！")
    st.sidebar.success("单击「预测医疗费用」进入功能页")
    st.markdown("""
    ## 应用说明
    该应用基于机器学习的随机森林回归算法，通过分析被保险人的个人信息，预测其未来医疗费用支出，为保险公司的保险定价提供参考依据。
    
    ### 核心功能
    - **费用预测**：输入被保险人相关信息，快速获取医疗费用预测结果
    - **数据支持**：基于真实医疗费用数据训练，预测结果具备参考价值
    
    ### 注意事项
    - 请输入准确、完整的信息，以提高预测结果的准确性
    - 预测结果仅作为保险定价参考，实际定价需结合更多业务因素审慎决策
    - 技术支持：<support@example.com>
    """)

# 定义预测页面函数
def predict_page():
    """当选择预测费用页面时，呈现该函数的内容"""
    st.write("# 医疗费用预测")
    st.markdown("### 请输入被保险人信息")
    
    # 创建用户输入表单
    with st.form('user_inputs'):
        # 数值型输入
        age = st.number_input('年龄', min_value=0, max_value=120, value=30, step=1)
        bmi = st.number_input('BMI（身体质量指数）', min_value=0.0, max_value=100.0, value=22.0, step=0.1)
        children = st.number_input('子女数量', min_value=0, max_value=10, value=0, step=1)
        
        # 分类型输入
        sex = st.radio('性别', options=['男性', '女性'])
        smoke = st.radio('是否吸烟', options=['是', '否'])
        region = st.selectbox('所在区域', options=['东南部', '西南部', '东北部', '西北部'])
        
        # 提交按钮
        submitted = st.form_submit_button('预测费用')
    
    # 表单提交后的处理逻辑
    if submitted:
        # 1. 初始化编码变量（分类特征二进制编码）
        sex_female, sex_male = 0, 0
        smoke_yes, smoke_no = 0, 0
        region_northeast, region_southeast, region_northwest, region_southwest = 0, 0, 0, 0
        
        # 2. 根据用户输入赋值编码变量
        # 性别编码
        if sex == '女性':
            sex_female = 1
        else:
            sex_male = 1
        
        # 吸烟状态编码
        if smoke == '是':
            smoke_yes = 1
        else:
            smoke_no = 1
        
        # 区域编码
        if region == '东北部':
            region_northeast = 1
        elif region == '东南部':
            region_southeast = 1
        elif region == '西北部':
            region_northwest = 1
        elif region == '西南部':
            region_southwest = 1
        
        # 3. 格式化输入数据（与模型训练时的特征顺序一致）
        format_data = [
            age, bmi, children, sex_female, sex_male,
            smoke_no, smoke_yes, region_northeast, region_southeast,
            region_northwest, region_southwest
        ]
        
        # 4. 加载预训练的随机森林回归模型
        try:
            with open('rfr_model.pkl', 'rb') as f:
                rfr_model = pickle.load(f)
            
            # 5. 转换数据格式为DataFrame（匹配模型输入要求）
            format_data_df = pd.DataFrame(
                data=[format_data],
                columns=rfr_model.feature_names_in_
            )
            
            # 6. 执行预测
            predict_result = rfr_model.predict(format_data_df)[0]
            
            # 7. 展示预测结果
            st.success(f'### 预测结果\n根据您输入的数据，该客户的医疗费用预测为：{round(predict_result, 2)} 元')
        
        except FileNotFoundError:
            st.error("错误：未找到模型文件 'rfr_model.pkl'，请确保模型文件与代码在同一目录下")
        except Exception as e:
            st.error(f"预测过程出错：{str(e)}")
    
    st.markdown("---")
    st.write("技术支持：<support@example.com>")

# 侧边栏导航
nav = st.sidebar.radio("导航", ["简介", "预测医疗费用"])

# 根据导航选择展示对应页面
if nav == "简介":
    introduce_page()
else:
    predict_page()
