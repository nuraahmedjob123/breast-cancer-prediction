import pandas as pd
import joblib
from flask import Flask, render_template, request
import numpy as np

# تهيئة تطبيق Flask
app = Flask(__name__)

# المسار إلى ملف النموذج
MODEL_PATH = 'cancer_prediction_model.pkl'

# تحميل النموذج المُدرب عند بدء تشغيل التطبيق
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print("✅ تم تحميل النموذج بنجاح.")
except FileNotFoundError:
    print(f"❌ لم يتم العثور على ملف النموذج: {MODEL_PATH}")
    model_pipeline = None
    
# قائمة أسماء الأعمدة المستخدمة للتدريب
FEATURE_NAMES = [
    "Clump_Thickness", "Uniformity_of_Cell_Size",
    "Uniformity_of_Cell_Shape", "Marginal_Adhesion", "Single_Epithelial_Cell_Size",
    "Bare_Nuclei", "Bland_Chromatin", "Normal_Nucleoli", "Mitoses",
]

# المسار الرئيسي (لعرض الواجهة الترحيبية)
@app.route('/')
def welcome():
    """يعرض صفحة الترحيب التي تحمل اسم المالك."""
    # يجب أن يكون ملف 'welcome.html' داخل مجلد 'templates'
    return render_template('welcome.html')

# المسار الخاص بنموذج التنبؤ (يستقبل GET لعرض النموذج و POST لإجراء التنبؤ)
@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    """
    إذا كانت الطريقة GET: يعرض صفحة نموذج إدخال البيانات (index.html).
    إذا كانت الطريقة POST: يستقبل البيانات ويُجري التنبؤ.
    """
    
    # ------------------ (GET) عرض النموذج ------------------
    if request.method == 'GET':
        # يعرض نموذج الإدخال لأول مرة
        return render_template('index.html', prediction_text=None)

    # ------------------ (POST) إجراء التنبؤ ------------------
    elif request.method == 'POST':
        if model_pipeline is None:
            return render_template('index.html', prediction_text="خطأ: لم يتم تحميل النموذج."), 500
        
        try:
            # (1) جمع البيانات
            data = request.form.to_dict()
            
            # (2) تحويل البيانات
            input_data = [int(data[col]) for col in FEATURE_NAMES]
            df_input = pd.DataFrame([input_data], columns=FEATURE_NAMES)
            
            # (3) إجراء التنبؤ
            prediction = model_pipeline.predict(df_input)[0]
            
            # (4) تفسير النتيجة
            if prediction == 1:
                message = "⚠️ بناءً على البيانات، هناك احتمال كبير بأن تكون الخلايا خبيثة. يرجى مراجعة طبيب مختص."
            else:
                message = "✅ بناءً على البيانات، الخلايا تبدو حميدة."
            
            # (5) إرجاع النتيجة إلى نفس صفحة index.html
            return render_template('index.html', prediction_text=message)

        except Exception as e:
            error_message = f"حدث خطأ أثناء التنبؤ. تأكد من إدخال جميع القيم: {e}"
            print(error_message)
            return render_template('index.html', prediction_text=f"خطأ في الإدخال: {e}"), 400

if __name__ == "__main__":
    app.run(debug=True)