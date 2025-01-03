import streamlit as st
import pandas as pd
import base64, time, spacy, io, random, re
from streamlit_tags import st_tags
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos

# Load spaCy model
import spacy
from spacy.cli import download

# try:
#     nlp = spacy.load('en_core_web_sm')
# except OSError:
#     download('en_core_web_sm')
#     nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en_core_web_sm')

class ResumeParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = self._extract_text()
        self.doc = nlp(self.text)
        
    def _extract_text(self):
        with open(self.pdf_path, 'rb') as fh:
            resource_mgr = PDFResourceManager()
            output = io.StringIO()
            converter = TextConverter(resource_mgr, output, laparams=LAParams())
            interpreter = PDFPageInterpreter(resource_mgr, converter)
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                interpreter.process_page(page)
            text = output.getvalue()
            converter.close()
            output.close()
            return text

    def get_email(self):
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, self.text)
        return emails[0] if emails else None

    def get_phone(self):
        phone_pattern = r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'
        phones = re.findall(phone_pattern, self.text)
        return phones[0] if phones else None

    def get_name(self):
        for ent in self.doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return None

    def get_skills(self):
        skills_db = {
            'languages': ['python', 'java', 'c++', 'javascript', 'ruby', 'golang'],
            'frontend': ['react', 'angular', 'vue', 'html', 'css', 'bootstrap'],
            'backend': ['node', 'django', 'flask', 'spring', 'express'],
            'database': ['sql', 'mongodb', 'postgresql', 'mysql', 'redis'],
            'devops': ['docker', 'kubernetes', 'jenkins', 'aws', 'azure', 'gcp'],
            'ml_ai': ['tensorflow', 'pytorch', 'scikit-learn', 'opencv', 'nlp'],
            'tools': ['git', 'jira', 'confluence', 'slack', 'trello']
        }
        
        found_skills = {category: [] for category in skills_db}
        for category, skills in skills_db.items():
            for skill in skills:
                if re.search(r'\b' + skill + r'\b', self.text.lower()):
                    found_skills[category].append(skill)
        return found_skills

    def get_education(self):
        edu_patterns = ['bachelor', 'master', 'phd', 'BTech','Btech','b.tech', 'm.tech', 'b.e', 'm.e']
        education = []
        for sent in self.doc.sents:
            if any(edu in sent.text.lower() for edu in edu_patterns):
                education.append(sent.text.strip())
        return education

    def analyze(self):
        return {
            'name': self.get_name(),
            'email': self.get_email(),
            'phone': self.get_phone(),
            'skills': self.get_skills(),
            'education': self.get_education(),
            'no_of_pages': len(list(PDFPage.get_pages(open(self.pdf_path, 'rb'))))
        }

class ResumeAnalyzer:
    def __init__(self):
        self.setup_page()
        
    def setup_page(self):
        st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
        
        # Custom CSS
        st.markdown("""
        <style>
        .main { padding: 2rem; }
        .stProgress > div > div > div > div { background-color: #00c853; }
        .css-1v0mbdj { padding-top: 0; }
        .css-10trblm { color: #1e88e5; }
        .highlight { 
            background-color: #f5f5f5; 
            padding: 1.5rem; 
            border-radius: 0.5rem; 
            margin: 1rem 0;
            border-left: 5px solid #1e88e5;
        }
        .skill-tag {
            display: inline-block;
            background-color: #e3f2fd;
            padding: 0.3rem 0.8rem;
            margin: 0.2rem;
            border-radius: 1rem;
            font-size: 0.9rem;
        }
        </style>
        """, unsafe_allow_html=True)

    def create_header(self):
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image('Logo/logo2.png', width=300)
            st.title("AI Resume Analyzer")
            st.markdown("""
            <div style='text-align: center'>
                <p>Upload your resume and get instant insights and recommendations</p>
            </div>
            """, unsafe_allow_html=True)

    def show_pdf(self, file_path):
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

    def create_skill_chart(self, skills):
        categories = []
        counts = []
        for category, skill_list in skills.items():
            if skill_list:
                categories.append(category.replace('_', ' ').title())
                counts.append(len(skill_list))
        
        if categories:
            fig = go.Figure(data=[go.Pie(labels=categories, values=counts, hole=.3)])
            fig.update_layout(title_text="Skills Distribution")
            st.plotly_chart(fig)

    def display_skills(self, skills):
        for category, skill_list in skills.items():
            if skill_list:
                st.markdown(f"**{category.replace('_', ' ').title()}**")
                for skill in skill_list:
                    st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)

    def calculate_score(self, resume_data):
        score = 0
        max_scores = {
            'contact_info': 20,
            'skills_diversity': 30,
            'education': 20,
            'length': 15,
            'skill_relevance': 15
        }
        
        # Contact info score
        if resume_data['email'] and resume_data['phone']:
            score += max_scores['contact_info']
        
        # Skills diversity score
        skill_categories = sum(1 for skills in resume_data['skills'].values() if skills)
        score += min(skill_categories * 5, max_scores['skills_diversity'])
        
        # Education score
        if resume_data['education']:
            score += max_scores['education']
        
        # Length score
        if resume_data['no_of_pages'] >= 2:
            score += max_scores['length']
        
        # Skill relevance score
        total_skills = sum(len(skills) for skills in resume_data['skills'].values())
        if total_skills >= 10:
            score += max_scores['skill_relevance']
        
        return score

    def recommend_courses(self, skills):
        all_skills = [skill for skill_list in skills.values() for skill in skill_list]
        
        tech_stacks = {
            'Data Science & ML': (['python', 'tensorflow', 'pytorch', 'scikit-learn'], ds_course),
            'Web Development': (['javascript', 'react', 'node', 'django'], web_course),
            'Android Development': (['java', 'kotlin', 'android'], android_course),
            'iOS Development': (['swift', 'objective-c'], ios_course),
            'UI/UX Design': (['figma', 'sketch', 'adobe'], uiux_course)
        }

        st.markdown("### 📚 Learning Path Recommendations")
        for stack_name, (keywords, courses) in tech_stacks.items():
            if any(skill in keywords for skill in all_skills):
                with st.expander(f"{stack_name} Path"):
                    random.shuffle(courses)
                    for idx, (course, link) in enumerate(courses[:3], 1):
                        st.markdown(f"{idx}. [{course}]({link})")

    def run(self):
        self.create_header()
        
        uploaded_file = st.file_uploader("Choose your Resume (PDF)", type="pdf")
        if uploaded_file:
            save_path = f"./Uploaded_Resumes/{uploaded_file.name}"
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("🔍 Analyzing your resume..."):
                parser = ResumeParser(save_path)
                resume_data = parser.analyze()
                time.sleep(1)
            
            tab1, tab2, tab3 = st.tabs(["📄 Resume", "📊 Analysis", "🎯 Recommendations"])
            
            with tab1:
                self.show_pdf(save_path)
            
            with tab2:
                col1, col2 = st.columns([2,1])
                
                with col1:
                    st.markdown("### 👤 Basic Information")
                    with st.container():
                        st.markdown(f"""
                        <div class='highlight'>
                        {'<p><strong>Name:</strong> ' + resume_data['name'] + '</p>' if resume_data['name'] else ''}
                        <p><strong>Email:</strong> {resume_data['email'] or 'Not found'}</p>
                        <p><strong>Phone:</strong> {resume_data['phone'] or 'Not found'}</p>
                        <p><strong>Pages:</strong> {resume_data['no_of_pages']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("### 🔧 Skills Overview")
                    self.create_skill_chart(resume_data['skills'])
                    self.display_skills(resume_data['skills'])
                    
                    st.markdown("### 🎓 Education")
                    for edu in resume_data['education']:
                        st.markdown(f"- {edu}")
                
                with col2:
                    st.markdown("### 📊 Resume Score")
                    score = self.calculate_score(resume_data)
                    st.markdown(f"""
                    <div style='text-align: center'>
                        <h1 style='font-size: 4rem; color: {'#00c853' if score >= 80 else '#ffd600' if score >= 60 else '#ff3d00'}'>{score}%</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(score/100)
                    
                    if score < 60:
                        st.error("Your resume needs significant improvement!")
                    elif score < 80:
                        st.warning("Your resume is good but has room for improvement.")
                    else:
                        st.success("Excellent resume! You're ready to apply.")
            
            with tab3:
                self.recommend_courses(resume_data['skills'])
                
                st.markdown("### 🎥 Helpful Resources")
                col1, col2 = st.columns(2)
                with col1:
                    st.video(random.choice(resume_videos))
                with col2:
                    st.video(random.choice(interview_videos))

if __name__ == "__main__":
    analyzer = ResumeAnalyzer()
    analyzer.run()
