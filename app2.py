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
import numpy as np


from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load spaCy model with error handling
# os.environ['SPACY_MODELS'] = os.path.expanduser('~/.local/lib/python3.12/site-packages/en_core_web_sm')
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')

nlp = spacy.load("en_core_web_sm")

class EnhancedResumeParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = self._extract_text()
        self.doc = nlp(self.text)
        self.sections = self._split_into_sections()
        
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

    def _split_into_sections(self):
        # Common section headers in resumes
        section_headers = [
            "education", "experience", "skills", "projects", 
            "certifications", "publications", "achievements",
            "work experience", "professional experience"
        ]
        
        sections = {}
        current_section = "other"
        current_text = []
        
        for line in self.text.split('\n'):
            line = line.strip().lower()
            if any(header in line for header in section_headers):
                if current_text:
                    sections[current_section] = '\n'.join(current_text)
                current_section = next(header for header in section_headers if header in line)
                current_text = []
            else:
                current_text.append(line)
                
        if current_text:
            sections[current_section] = '\n'.join(current_text)
            
        return sections

    def get_email(self):
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, self.text)
        return emails[0] if emails else None

    def get_phone(self):
        phone_pattern = r'(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}|\(\d{3}\)\s*\d{3}[-.\s]?\d{4}|\d{3}[-.\s]?\d{4})'
        phones = re.findall(phone_pattern, self.text)
        return phones[0] if phones else None

    def get_name(self):
        first_paragraph = self.text.split('\n')[0]
        doc = nlp(first_paragraph)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return None

    def get_skills(self):
        skills_db = {
            'programming': [
                'python', 'java', 'c++', 'javascript', 'typescript', 'ruby', 'golang',
                'php', 'swift', 'kotlin', 'rust', 'scala', 'r', 'matlab'
            ],
            'frontend': [
                'react', 'angular', 'vue', 'html', 'css', 'bootstrap', 'jquery',
                'sass', 'less', 'webpack', 'babel', 'typescript', 'next.js', 'gatsby'
            ],
            'backend': [
                'node', 'django', 'flask', 'spring', 'express', 'fastapi', 'ruby on rails',
                'laravel', 'asp.net', 'graphql', 'rest api', 'microservices'
            ],
            'database': [
                'sql', 'mongodb', 'postgresql', 'mysql', 'redis', 'oracle', 'cassandra',
                'elasticsearch', 'dynamodb', 'firebase', 'neo4j'
            ],
            'devops': [
                'docker', 'kubernetes', 'jenkins', 'aws', 'azure', 'gcp', 'terraform',
                'ansible', 'ci/cd', 'git', 'github actions', 'gitlab ci'
            ],
            'ml_ai': [
                'tensorflow', 'pytorch', 'scikit-learn', 'opencv', 'nlp', 'computer vision',
                'deep learning', 'machine learning', 'neural networks', 'data science'
            ],
            'cloud': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'serverless',
                'lambda', 'ec2', 's3', 'rds', 'cloudformation'
            ],
            'tools': [
                'git', 'jira', 'confluence', 'slack', 'trello', 'figma', 'sketch',
                'adobe xd', 'postman', 'swagger', 'linux', 'agile', 'scrum'
            ]
        }
        
        found_skills = {category: [] for category in skills_db}
        for category, skills in skills_db.items():
            for skill in skills:
                if re.search(r'\b' + re.escape(skill) + r'\b', self.text.lower()):
                    found_skills[category].append(skill)
        return found_skills

    def get_education(self):
        education = []
        edu_section = self.sections.get('education', '')
        if edu_section:
            doc = nlp(edu_section)
            for sent in doc.sents:
                if any(keyword in sent.text.lower() for keyword in [
                    'bachelor', 'master', 'phd', 'b.tech', 'm.tech', 'b.e', 'm.e',
                    'diploma', 'degree', 'university', 'college'
                ]):
                    education.append(sent.text.strip())
        return education

    def get_experience(self):
        experience = []
        exp_section = self.sections.get('experience', '') or self.sections.get('work experience', '')
        if exp_section:
            # Split into entries based on date patterns
            date_pattern = r'\b(19|20)\d{2}\b|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}'
            entries = re.split(date_pattern, exp_section)
            for entry in entries:
                if entry.strip():
                    experience.append(entry.strip())
        return experience

    def get_projects(self):
        projects = []
        proj_section = self.sections.get('projects', '')
        if proj_section:
            # Split by common project delimiters
            project_entries = re.split(r'\n(?=•|\*|\-|Project\s+\d+:)', proj_section)
            for entry in project_entries:
                if entry.strip():
                    projects.append(entry.strip())
        return projects

    def analyze(self):
        return {
            'name': self.get_name(),
            'email': self.get_email(),
            'phone': self.get_phone(),
            'skills': self.get_skills(),
            'education': self.get_education(),
            'experience': self.get_experience(),
            'projects': self.get_projects(),
            'no_of_pages': len(list(PDFPage.get_pages(open(self.pdf_path, 'rb'))))
        }

class ModernResumeAnalyzer:
    def __init__(self):
        self.setup_page()
        
    def setup_page(self):
        st.set_page_config(
            page_title="Smart Resume Analyzer",
            page_icon="📑",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Modern CSS with animations and better formatting
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .main {
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .stProgress > div > div > div > div {
            background: linear-gradient(to right, #00c853, #64dd17);
        }
        
        .css-1v0mbdj {
            padding-top: 0;
        }
        
        .css-10trblm {
            color: #1e88e5;
            font-weight: 600;
        }
        
        .highlight {
            background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            margin: 1rem 0;
            border-left: 5px solid #1e88e5;
            transition: transform 0.2s;
        }
        
        .highlight:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .skill-tag {
            display: inline-block;
            background: linear-gradient(120deg, #e3f2fd 0%, #bbdefb 100%);
            padding: 0.4rem 1rem;
            margin: 0.3rem;
            border-radius: 2rem;
            font-size: 0.9rem;
            transition: all 0.2s;
            cursor: default;
        }
        
        .skill-tag:hover {
            transform: scale(1.05);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #1e88e5;
        }
        
        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 1rem 0;
            transition: transform 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        }
        
        .metric {
            text-align: center;
            padding: 1rem;
            background: linear-gradient(120deg, #f5f5f5 0%, #fafafa 100%);
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 600;
            color: #1e88e5;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)

    def create_header(self):
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image('Logo/logo2.png', use_container_width=True)
            st.title("🎯 Smart Resume Analyzer")
            st.markdown("""
            <div style='text-align: center'>
                <p style='font-size: 1.2rem; color: #666;'>
                    Upload your resume and get AI-powered insights, recommendations, and analytics
                </p>
            </div>
            """, unsafe_allow_html=True)

    def show_pdf(self, file_path):
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'''
            <iframe 
                src="data:application/pdf;base64,{base64_pdf}" 
                width="100%" 
                height="800" 
                style="border: none; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
            >
            </iframe>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)

    def create_skill_chart(self, skills):
        # Create a more sophisticated skills visualization
        categories = []
        counts = []
        all_skills = []
        
        for category, skill_list in skills.items():
            if skill_list:
                categories.append(category.replace('_', ' ').title())
                counts.append(len(skill_list))
                all_skills.extend(skill_list)
        
        if categories:
            # Create a sunburst chart
            fig = go.Figure(go.Sunburst(
                labels=categories + all_skills,
                parents=[''] * len(categories) + categories * (len(all_skills) // len(categories)),
                values=[10] * len(categories) + [5] * len(all_skills),
                branchvalues="total",
            ))
            
            fig.update_layout(
                title_text="Skills Distribution",
                width=600,
                height=600,
                template="plotly_white"
            )
            st.plotly_chart(fig)
            
            # Create word cloud of skills
            if all_skills:
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    colormap='viridis'
                ).generate(' '.join(all_skills))
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)

    def display_skills(self, skills):
        for category, skill_list in skills.items():
            if skill_list:
                st.markdown(f"<h4>{category.replace('_', ' ').title()}</h4>", unsafe_allow_html=True)
                for skill in skill_list:
                    st.markdown(
                        f'''
                        <span class="skill-tag" title="Click to learn more about {skill}">
                            {skill}
                        </span>
                        ''',
                        unsafe_allow_html=True
                    )
                    
                    
    def calculate_score(self, resume_data):
        score = 0
        max_scores = {
            'contact_info': 15,
            'skills_diversity': 25,
            'education': 15,
            'experience': 20,
            'projects': 15,
            'length': 10
        }
        
        # Contact info score
        if resume_data['email']:
            score += 7.5
        if resume_data['phone']:
            score += 7.5
        
        # Skills diversity score
        skill_categories = sum(1 for skills in resume_data['skills'].values() if skills)
        total_skills = sum(len(skills) for skills in resume_data['skills'].values())
        score += min((skill_categories * 5) + (total_skills * 0.5), max_scores['skills_diversity'])
        
        # Education score
        if resume_data['education']:
            education_keywords = ['bachelor', 'master', 'phd', 'diploma']
            edu_score = sum(2 for edu in resume_data['education'] 
                        if any(keyword in edu.lower() for keyword in education_keywords))
            score += min(edu_score, max_scores['education'])
        
        # Experience score
        if resume_data['experience']:
            score += min(len(resume_data['experience']) * 5, max_scores['experience'])
        
        # Projects score
        if resume_data['projects']:
            score += min(len(resume_data['projects']) * 3, max_scores['projects'])
        
        # Length score
        if resume_data['no_of_pages'] >= 1 and resume_data['no_of_pages'] <= 3:
            score += max_scores['length']
        elif resume_data['no_of_pages'] > 3:
            score += max_scores['length'] * 0.5
        
        return score

    def create_radar_chart(self, resume_data):
        # Calculate individual section scores
        categories = ['Contact Info', 'Skills', 'Education', 'Experience', 'Projects', 'Format']
        scores = [
            # Contact Info (out of 100)
            (100 if resume_data['email'] and resume_data['phone'] else 
             50 if resume_data['email'] or resume_data['phone'] else 0),
            
            # Skills (out of 100)
            min(100, sum(len(skills) for skills in resume_data['skills'].values()) * 10),
            
            # Education (out of 100)
            min(100, len(resume_data['education']) * 25),
            
            # Experience (out of 100)
            min(100, len(resume_data['experience']) * 20),
            
            # Projects (out of 100)
            min(100, len(resume_data['projects']) * 25),
            
            # Format (out of 100)
            100 if 1 <= resume_data['no_of_pages'] <= 2 else 
            80 if resume_data['no_of_pages'] == 3 else 50
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='Your Resume'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False
        )
        
        return fig

    def display_improvement_suggestions(self, resume_data):
        suggestions = []
        
        # Contact information suggestions
        if not resume_data['email'] or not resume_data['phone']:
            suggestions.append({
                'category': 'Contact Information',
                'priority': 'High',
                'suggestion': 'Add both email and phone number for better reachability.'
            })
            
        # Skills suggestions
        total_skills = sum(len(skills) for skills in resume_data['skills'].values())
        if total_skills < 10:
            suggestions.append({
                'category': 'Skills',
                'priority': 'High',
                'suggestion': 'Include more relevant technical and soft skills. Aim for at least 10-15 key skills.'
            })
            
        # Education suggestions
        if not resume_data['education']:
            suggestions.append({
                'category': 'Education',
                'priority': 'High',
                'suggestion': 'Add your educational background, including degrees and relevant coursework.'
            })
            
        # Experience suggestions
        if not resume_data['experience']:
            suggestions.append({
                'category': 'Experience',
                'priority': 'High',
                'suggestion': 'Add your work experience with detailed responsibilities and achievements.'
            })
        elif len(resume_data['experience']) < 3:
            suggestions.append({
                'category': 'Experience',
                'priority': 'Medium',
                'suggestion': 'Consider adding more details about your work experiences, including metrics and achievements.'
            })
            
        # Projects suggestions
        if not resume_data['projects']:
            suggestions.append({
                'category': 'Projects',
                'priority': 'Medium',
                'suggestion': 'Include relevant projects to showcase your practical skills.'
            })
            
        # Length suggestions
        if resume_data['no_of_pages'] > 3:
            suggestions.append({
                'category': 'Format',
                'priority': 'Medium',
                'suggestion': 'Consider condensing your resume to 2-3 pages for better readability.'
            })
            
        return suggestions

    def recommend_courses(self, skills, experience):
        all_skills = [skill for skill_list in skills.values() for skill in skill_list]
        experience_text = ' '.join(experience)
        
        career_paths = {
            'Data Science & ML': {
                'keywords': ['python', 'tensorflow', 'pytorch', 'scikit-learn', 'data', 'machine learning'],
                'courses': ds_course,
                'icon': '📊'
            },
            'Web Development': {
                'keywords': ['javascript', 'react', 'node', 'django', 'html', 'css'],
                'courses': web_course,
                'icon': '🌐'
            },
            'Android Development': {
                'keywords': ['java', 'kotlin', 'android'],
                'courses': android_course,
                'icon': '📱'
            },
            'iOS Development': {
                'keywords': ['swift', 'objective-c', 'ios'],
                'courses': ios_course,
                'icon': '🍎'
            },
            'UI/UX Design': {
                'keywords': ['figma', 'sketch', 'adobe', 'design', 'user experience'],
                'courses': uiux_course,
                'icon': '🎨'
            }
        }

        st.markdown("### 🎯 Personalized Learning Paths")
        
        for path_name, path_info in career_paths.items():
            matching_keywords = sum(1 for keyword in path_info['keywords'] 
                                 if keyword in ' '.join(all_skills).lower() 
                                 or keyword in experience_text.lower())
            
            if matching_keywords >= 2:  # Show path if at least 2 keywords match
                with st.expander(f"{path_info['icon']} {path_name} Path"):
                    st.markdown("""
                    <div class="card">
                        <h4>Recommended Courses:</h4>
                    """, unsafe_allow_html=True)
                    
                    courses = path_info['courses']
                    random.shuffle(courses)
                    for idx, (course, link) in enumerate(courses[:3], 1):
                        st.markdown(f"{idx}. [{course}]({link})")
                    
                    st.markdown("</div>", unsafe_allow_html=True)

    def run(self):
        self.create_header()
        
        uploaded_file = st.file_uploader("Choose your Resume (PDF)", type="pdf")
        if uploaded_file:
            save_path = f"./Uploaded_Resumes/{uploaded_file.name}"
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("🤖 Analyzing your resume with AI..."):
                parser = EnhancedResumeParser(save_path)
                resume_data = parser.analyze()
                time.sleep(1)
            
            tabs = st.tabs(["📄 Resume", "📊 Analysis", "💡 Insights", "🎯 Recommendations"])
            
            with tabs[0]:  # Resume Tab
                self.show_pdf(save_path)
            
            with tabs[1]:  # Analysis Tab
                col1, col2 = st.columns([2,1])
                
                with col1:
                    st.markdown("### 👤 Professional Profile")
                    with st.container():
                        st.markdown(f"""
                        <div class='highlight'>
                            {'<p><strong>Name:</strong> ' + resume_data['name'] + '</p>' if resume_data['name'] else ''}
                            <p><strong>Email:</strong> {resume_data['email'] or 'Not found'}</p>
                            <p><strong>Phone:</strong> {resume_data['phone'] or 'Not found'}</p>
                            <p><strong>Resume Length:</strong> {resume_data['no_of_pages']} page(s)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("### 🔧 Skills Analysis")
                    self.create_skill_chart(resume_data['skills'])
                    self.display_skills(resume_data['skills'])
                
                with col2:
                    st.markdown("### 📊 Resume Score")
                    score = self.calculate_score(resume_data)
                    
                    # Display score with animation
                    st.markdown(f"""
                    <div class="card" style="text-align: center;">
                        <div class="metric">
                            <div class="metric-value" style="color: {'#00c853' if score >= 80 else '#ffd600' if score >= 60 else '#ff3d00'}">
                                {score}%
                            </div>
                            <div class="metric-label">Overall Score</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.plotly_chart(self.create_radar_chart(resume_data))
            
            with tabs[2]:  # Insights Tab
                st.markdown("### 🎓 Education")
                if resume_data['education']:
                    for edu in resume_data['education']:
                        st.markdown(f"<div class='card'>{edu}</div>", unsafe_allow_html=True)
                else:
                    st.warning("No education information found.")
                
                st.markdown("### 💼 Professional Experience")
                if resume_data['experience']:
                    for exp in resume_data['experience']:
                        st.markdown(f"<div class='card'>{exp}</div>", unsafe_allow_html=True)
                else:
                    st.warning("No professional experience found.")
                
                st.markdown("### 🚀 Projects")
                if resume_data['projects']:
                    for project in resume_data['projects']:
                        st.markdown(f"<div class='card'>{project}</div>", unsafe_allow_html=True)
                else:
                    st.warning("No projects found.")
            
            with tabs[3]:  # Recommendations Tab
                col1, col2 = st.columns([2,1])
                
                with col1:
                    st.markdown("### 📈 Improvement Suggestions")
                    suggestions = self.display_improvement_suggestions(resume_data)
                    for suggestion in suggestions:
                        st.markdown(f"""
                        <div class="card">
                            <h4>{suggestion['category']}</h4>
                            <p style="color: {'#ff3d00' if suggestion['priority'] == 'High' else '#ffd600'}">
                                Priority: {suggestion['priority']}
                            </p>
                            <p>{suggestion['suggestion']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    self.recommend_courses(resume_data['skills'], resume_data['experience'])
                    
                    st.markdown("### 🎥 Career Resources")
                    st.video(random.choice(resume_videos))
                    st.video(random.choice(interview_videos))

if __name__ == "__main__":
    analyzer = ModernResumeAnalyzer()
    analyzer.run()
