import streamlit as st
import re
import io
import base64
import time
import random
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter

class EnhancedResumeParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = self._extract_text()
        
    def _extract_text(self):
        text = ""
        with open(self.pdf_path, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                resource_mgr = PDFResourceManager()
                output = io.StringIO()
                converter = TextConverter(resource_mgr, output, laparams=LAParams())
                interpreter = PDFPageInterpreter(resource_mgr, converter)
                interpreter.process_page(page)
                text += output.getvalue()
                converter.close()
                output.close()
        return text

    def get_basic_info(self):
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        
        emails = re.findall(email_pattern, self.text)
        phones = re.findall(phone_pattern, self.text)
        linkedin = re.findall(linkedin_pattern, self.text.lower())
        
        return {
            'name': self.text.split('\n')[0].strip(),
            'email': emails[0] if emails else None,
            'phone': phones[0] if phones else None,
            'linkedin': linkedin[0] if linkedin else None,
            'no_of_pages': len(list(PDFPage.get_pages(open(self.pdf_path, 'rb'))))
        }

    def get_skills(self):
        skill_categories = {
            'Programming Languages': [
                'python', 'java', 'javascript', 'c++', 'ruby', 'php',
                'swift', 'typescript', 'golang', 'rust', 'r', 'matlab'
            ],
            'Web Development': [
                'html', 'css', 'react', 'angular', 'vue', 'node.js',
                'django', 'flask', 'bootstrap', 'jquery', 'sass', 'webpack'
            ],
            'Databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'oracle',
                'redis', 'elasticsearch', 'firebase', 'dynamodb'
            ],
            'Cloud & DevOps': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
                'terraform', 'ci/cd', 'git', 'github actions'
            ],
            'AI & ML': [
                'machine learning', 'deep learning', 'tensorflow', 'pytorch',
                'scikit-learn', 'nlp', 'computer vision', 'neural networks'
            ],
            'Soft Skills': [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'project management', 'agile', 'scrum', 'presentation'
            ]
        }
        
        found_skills = {category: [] for category in skill_categories}
        text_lower = self.text.lower()
        
        for category, skills in skill_categories.items():
            for skill in skills:
                if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                    found_skills[category].append(skill)
        
        return found_skills

    def get_education(self):
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 
                            'college', 'diploma', 'certification']
        education = []
        
        lines = self.text.split('\n')
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in education_keywords):
                # Get the current line and the next line for context
                edu_info = ' '.join([line.strip(), lines[i+1].strip() if i+1 < len(lines) else '']).strip()
                education.append(edu_info)
        
        return education

    def calculate_ats_score(self):
        scores = {
            'contact_info': self._score_contact_info(),
            'skills': self._score_skills(),
            'education': self._score_education(),
            'format': self._score_format()
        }
        
        weights = {
            'contact_info': 0.25,
            'skills': 0.35,
            'education': 0.25,
            'format': 0.15
        }
        
        final_score = sum(scores[k] * weights[k] for k in scores)
        return scores, final_score

    def _score_contact_info(self):
        score = 0
        basic_info = self.get_basic_info()
        if basic_info['email']: score += 33.33
        if basic_info['phone']: score += 33.33
        if basic_info['linkedin']: score += 33.34
        return score

    def _score_skills(self):
        skills = self.get_skills()
        total_skills = sum(len(skill_list) for skill_list in skills.values())
        return min(100, total_skills * 5)  # 20 skills for max score

    def _score_education(self):
        education = self.get_education()
        return min(100, len(education) * 50)  # 2 education entries for max score

    def _score_format(self):
        pages = self.get_basic_info()['no_of_pages']
        if pages == 1: return 100
        elif pages == 2: return 90
        elif pages == 3: return 70
        else: return 50

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Smart Resume Analyzer", page_icon="📊", layout="wide")
    
    # Custom CSS with animations and modern design
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .css-1v0mbdj {
            padding-top: 0;
        }
        
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #00c6ff, #0072ff);
        }
        
        .skill-box {
            background: linear-gradient(135deg, #f6f9fc 0%, #f0f4f8 100%);
            padding: 10px;
            border-radius: 10px;
            margin: 5px;
            border: 1px solid #e0e0e0;
            transition: all 0.3s ease;
        }
        
        .skill-box:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .info-box {
            background: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
        }
        
        .score-card {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .score-value {
            font-size: 48px;
            font-weight: bold;
            background: linear-gradient(45deg, #00c6ff, #0072ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .recommendation-card {
            background: #fff;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #0072ff;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .recommendation-card:hover {
            transform: translateX(5px);
        }
        
        .section-title {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #1a1a1a;
            border-bottom: 2px solid #0072ff;
            padding-bottom: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("🎯 Smart Resume Analyzer")
        st.markdown("<p style='text-align: center; color: #666;'>Unlock the potential of your resume with AI-powered analysis</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf")
    
    if uploaded_file:
        save_path = f"temp_resume_{uploaded_file.name}"
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("🔍 Analyzing your resume..."):
            parser = EnhancedResumeParser(save_path)
            basic_info = parser.get_basic_info()
            skills = parser.get_skills()
            education = parser.get_education()
            section_scores, ats_score = parser.calculate_ats_score()
            time.sleep(1)  # For effect
        
        tabs = st.tabs(["📄 Resume", "📊 Analysis", "💡 Insights"])
        
        with tabs[0]:
            show_pdf(save_path)
        
        with tabs[1]:
            col1, col2 = st.columns([2,1])
            
            with col1:
                st.markdown("<div class='section-title'>👤 Professional Profile</div>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class='info-box'>
                        <p><strong>Name:</strong> {basic_info['name']}</p>
                        <p><strong>Email:</strong> {basic_info['email'] or 'Not found'}</p>
                        <p><strong>Phone:</strong> {basic_info['phone'] or 'Not found'}</p>
                        <p><strong>LinkedIn:</strong> {basic_info['linkedin'] or 'Not found'}</p>
                        <p><strong>Resume Length:</strong> {basic_info['no_of_pages']} page(s)</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<div class='section-title'>🔧 Skills Analysis</div>", unsafe_allow_html=True)
                for category, category_skills in skills.items():
                    if category_skills:
                        st.write(f"**{category}**")
                        cols = st.columns(3)
                        for idx, skill in enumerate(category_skills):
                            cols[idx % 3].markdown(f'<div class="skill-box">{skill}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='section-title'>📊 ATS Score</div>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class='score-card'>
                        <div class='score-value'>{ats_score:.1f}%</div>
                        <p>Overall ATS Score</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Section Scores
                for section, score in section_scores.items():
                    st.progress(score/100, f"{section.title()}: {score:.1f}%")
        
        with tabs[2]:
            col1, col2 = st.columns([2,1])
            
            with col1:
                st.markdown("<div class='section-title'>🎓 Education</div>", unsafe_allow_html=True)
                if education:
                    for edu in education:
                        st.markdown(f'<div class="info-box">{edu}</div>', unsafe_allow_html=True)
                else:
                    st.warning("No education information found")
                
                st.markdown("<div class='section-title'>💡 Recommendations</div>", unsafe_allow_html=True)
                recommendations = []
                
                if section_scores['contact_info'] < 100:
                    missing = []
                    if not basic_info['email']: missing.append("email")
                    if not basic_info['phone']: missing.append("phone")
                    if not basic_info['linkedin']: missing.append("LinkedIn")
                    recommendations.append(f"Add missing contact information: {', '.join(missing)}")
                
                if section_scores['skills'] < 70:
                    recommendations.append("Add more relevant skills to strengthen your profile")
                
                if section_scores['education'] < 50:
                    recommendations.append("Include more detailed education information")
                
                if basic_info['no_of_pages'] > 2:
                    recommendations.append("Consider condensing your resume to 1-2 pages")
                
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"""
                        <div class="recommendation-card">
                            <strong>#{i}</strong> {rec}
                        </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='section-title'>📈 Industry Stats</div>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class='info-box'>
                        <p>⚡ <strong>Top Skills in Demand:</strong></p>
                        <ul>
                            <li>Python</li>
                            <li>Machine Learning</li>
                            <li>Cloud Computing</li>
                            <li>Data Analysis</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()