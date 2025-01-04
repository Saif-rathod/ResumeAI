
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
from fpdf import FPDF
import json
from collections import Counter
import textwrap
from datetime import datetime
import string
from difflib import SequenceMatcher

nlp = spacy.load('en_core_web_sm')

class ATSOptimizer:
    def __init__(self):
        self.job_market_skills = {
            'software_development': [
                'python', 'java', 'javascript', 'react', 'angular', 'vue.js', 'node.js',
                'typescript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go',
                'rust', 'scala', 'perl', 'haskell', 'lua', 'r', 'matlab'
            ],
            'web_technologies': [
                'html5', 'css3', 'sass', 'less', 'bootstrap', 'tailwind', 'jquery',
                'webpack', 'babel', 'next.js', 'gatsby', 'vue', 'nuxt.js', 'svelte',
                'web components', 'progressive web apps', 'web sockets'
            ],
            'cloud_devops': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab',
                'terraform', 'ansible', 'puppet', 'chef', 'prometheus', 'grafana',
                'elk stack', 'cloudformation', 'circleci', 'travis ci'
            ],
            'data_science': [
                'pandas', 'numpy', 'scipy', 'scikit-learn', 'tensorflow', 'pytorch',
                'keras', 'matplotlib', 'seaborn', 'plotly', 'tableau', 'power bi',
                'spark', 'hadoop', 'big data', 'machine learning', 'deep learning'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
                'oracle', 'sql server', 'sqlite', 'neo4j', 'dynamodb', 'couchbase',
                'firebase', 'mariadb', 'graphql', 'apache hbase'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem-solving',
                'critical thinking', 'time management', 'project management',
                'agile', 'scrum', 'presentation', 'negotiation', 'mentoring'
            ],
            'tools': [
                'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence',
                'slack', 'trello', 'asana', 'notion', 'figma', 'sketch',
                'adobe xd', 'invision', 'postman', 'swagger'
            ]
        }
       
    def analyze_text_similarity(self, text1, text2):
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def optimize_resume(self, resume_data, job_description=None):
        optimized_content = {
            'header': self._create_header(resume_data),
            'summary': self._create_summary(resume_data),
            'skills': self._optimize_skills(resume_data['skills'], job_description),
            'experience': self._optimize_experience(resume_data['experience'], job_description),
            'education': resume_data['education'],
            'projects': self._optimize_projects(resume_data['projects'], job_description)
        }
        return optimized_content
    
    def _create_header(self, resume_data):
        return {
            'name': resume_data['name'],
            'email': resume_data['email'],
            'phone': resume_data['phone'],
            'linkedin': self._extract_linkedin(resume_data.get('text', ''))
        }
    
    def _create_summary(self, resume_data):
        total_exp = len(resume_data['experience'])
        skills_summary = ', '.join(self._get_top_skills(resume_data['skills'], 5))
        return f"Experienced professional with {total_exp}+ years of expertise in {skills_summary}."
    
    def _optimize_skills(self, skills, job_description=None):
        optimized_skills = {}
        for category, skill_list in skills.items():
            market_skills = self.job_market_skills.get(category.lower(), [])
            if job_description:
                # Prioritize skills mentioned in job description
                skill_list.sort(key=lambda x: job_description.lower().count(x.lower()), reverse=True)
            optimized_skills[category] = list(set(skill_list) | 
                                        set(s for s in market_skills if any(self.analyze_text_similarity(s, sk) > 0.8 for sk in skill_list)))
        return optimized_skills
    
    def _optimize_experience(self, experience, job_description=None):
        optimized_exp = []
        action_verbs = ['led', 'developed', 'implemented', 'created', 'managed',
                    'improved', 'increased', 'reduced', 'streamlined', 'architected']
        
        for exp in experience:
            # Add action verbs if not present
            if not any(verb in exp.lower() for verb in action_verbs):
                exp = f"{random.choice(action_verbs).capitalize()} {exp}"
            
            # Add metrics if not present
            if not any(c.isdigit() for c in exp):
                exp += " Improved efficiency by 25% through process optimization."
            
            optimized_exp.append(exp)
        return optimized_exp

    def _optimize_projects(self, projects, job_description=None):
        optimized_projects = []
        for project in projects:
            # Add technical details if not present
            if not any(tech in project.lower() for tech in sum(self.job_market_skills.values(), [])):
                project += f" Utilized {', '.join(random.sample(sum(self.job_market_skills.values(), []), 3))}."
            optimized_projects.append(project)
        return optimized_projects
    
    def generate_pdf(self, optimized_content):
        pdf = FPDF()
        pdf.add_page()
        
        # Set fonts
        pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
        pdf.set_font('DejaVu', '', 12)
        
        # Header
        pdf.set_font('DejaVu', '', 16)
        pdf.cell(0, 10, optimized_content['header']['name'], ln=True, align='C')
        pdf.set_font('DejaVu', '', 10)
        pdf.cell(0, 5, f"{optimized_content['header']['email']} | {optimized_content['header']['phone']}", ln=True, align='C')
        
        # Summary
        pdf.ln(5)
        pdf.set_font('DejaVu', '', 12)
        pdf.cell(0, 10, 'Professional Summary', ln=True)
        pdf.set_font('DejaVu', '', 10)
        pdf.multi_cell(0, 5, optimized_content['summary'])
        
        # Skills
        pdf.ln(5)
        pdf.set_font('DejaVu', '', 12)
        pdf.cell(0, 10, 'Skills', ln=True)
        for category, skills in optimized_content['skills'].items():
            if skills:
                pdf.set_font('DejaVu', '', 10)
                pdf.cell(0, 5, f"{category.replace('_', ' ').title()}: {', '.join(skills)}", ln=True)
        
        # Experience
        pdf.ln(5)
        pdf.set_font('DejaVu', '', 12)
        pdf.cell(0, 10, 'Professional Experience', ln=True)
        for exp in optimized_content['experience']:
            pdf.set_font('DejaVu', '', 10)
            pdf.multi_cell(0, 5, f"• {exp}")
        
        return pdf

class EnhancedResumeParser:
    def get_insights(self):
        return {
            'keyword_density': self._analyze_keyword_density(),
            'readability_score': self._calculate_readability(),
            'experience_timeline': self._create_experience_timeline(),
            'skill_gaps': self._identify_skill_gaps(),
            'industry_alignment': self._analyze_industry_alignment()
        }
    
    def _analyze_keyword_density(self):
        words = self.text.lower().split()
        word_freq = Counter(words)
        total_words = len(words)
        return {word: count/total_words * 100 
                for word, count in word_freq.most_common(20)}
    
    def _calculate_readability(self):
        sentences = len(list(self.doc.sents))
        words = len(self.text.split())
        syllables = sum(len(word.text) / 3 for word in self.doc)  # Rough approximation
        
        if sentences == 0:
            return 0
            
        return 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    
    def _create_experience_timeline(self):
        timeline = []
        for exp in self.get_experience():
            # Extract dates using regex
            dates = re.findall(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4})', exp)
            if len(dates) >= 2:
                timeline.append({
                    'start': dates[0],
                    'end': dates[1],
                    'description': exp
                })
        return timeline
    
    def _identify_skill_gaps(self):
        current_skills = set(skill for skills in self.get_skills().values() for skill in skills)
        
        # Common skills by job role
        role_skills = {
            'Software Engineer': {'python', 'java', 'javascript', 'sql', 'git'},
            'Data Scientist': {'python', 'machine learning', 'sql', 'statistics', 'tensorflow'},
            'Web Developer': {'html', 'css', 'javascript', 'react', 'node.js'},
            'DevOps Engineer': {'docker', 'kubernetes', 'jenkins', 'aws', 'terraform'}
        }
        
        gaps = {}
        for role, required_skills in role_skills.items():
            missing_skills = required_skills - current_skills
            if missing_skills:
                gaps[role] = list(missing_skills)
        
        return gaps
    
    def _analyze_industry_alignment(self):
        # Industry keywords
        industries = {
            'Technology': ['software', 'technology', 'tech', 'digital', 'cloud', 'ai'],
            'Finance': ['banking', 'finance', 'investment', 'trading', 'financial'],
            'Healthcare': ['medical', 'healthcare', 'health', 'clinical', 'patient'],
            'E-commerce': ['retail', 'e-commerce', 'shopping', 'marketplace', 'consumer']
        }
        
        alignment = {}
        for industry, keywords in industries.items():
            score = sum(self.text.lower().count(keyword) for keyword in keywords)
            alignment[industry] = min(score * 10, 100)  # Scale score to 0-100
        
        return alignment

class ModernResumeAnalyzer:
    def __init__(self):
        super().__init__()
        self.optimizer = ATSOptimizer()
    
    def setup_page(self):
        
        st.set_page_config(
            page_title="Smart Resume Analyzer",
            page_icon="📑",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add to existing setup_page method
        st.markdown("""
        <style>
        /* Add these new styles */
        .insight-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-radius: 1rem;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .insight-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        .timeline-item {
            position: relative;
            padding-left: 2rem;
            margin: 1rem 0;
            border-left: 2px solid #1e88e5;
        }
        
        .timeline-item::before {
            content: '';
            position: absolute;
            left: -0.5rem;
            top: 0;
            width: 1rem;
            height: 1rem;
            background: #1e88e5;
            border-radius: 50%;
        }
        
        .tip-card {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #1e88e5;
        }
        
        .skill-progress {
            height: 0.5rem;
            background: #e0e0e0;
            border-radius: 0.25rem;
            margin: 0.25rem 0;
        }
        
        .skill-progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #1e88e5 0%, #64b5f6 100%);
            border-radius: 0.25rem;
            transition: width 0.5s ease;
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
        
    def display_insights(self, resume_data, insights):
        st.markdown("## 📊 Resume Insights")
        
        # Keyword Analysis
        with st.expander("📝 Keyword Analysis"):
            fig = go.Figure(data=[
                go.Bar(x=list(insights['keyword_density'].keys()),
                    y=list(insights['keyword_density'].values()))
            ])
            fig.update_layout(title="Keyword Density Analysis")
            st.plotly_chart(fig)
        
        # Experience Timeline
        with st.expander("⏳ Experience Timeline"):
            if insights['experience_timeline']:
                fig = go.Figure()
                for item in insights['experience_timeline']:
                    fig.add_trace(go.Scatter(
                        x=[item['start'], item['end']],
                        y=[random.random()],
                        mode="lines+markers",
                        name=textwrap.shorten(item['description'], width=50)
                    ))
                fig.update_layout(title="Career Timeline")
                st.plotly_chart(fig)
            else:
                st.warning("No timeline data available")
        
        # Industry Alignment
        with st.expander("🎯 Industry Alignment"):
            fig = go.Figure(data=[
                go.Scatterpolar(
                    r=list(insights['industry_alignment'].values()),
                    theta=list(insights['industry_alignment'].keys()),
                    fill='toself'
                )
            ])
            fig.update_layout(title="Industry Alignment Analysis")
            st.plotly_chart(fig)
        
        # Skill G
        
# Continuing from Skill Gaps section
        with st.expander("🎯 Skill Gaps Analysis"):
            if insights['skill_gaps']:
                st.markdown("### Role-Specific Skill Gaps")
                for role, missing_skills in insights['skill_gaps'].items():
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>{role}</h4>
                        <p>To be more competitive for this role, consider learning:</p>
                        {''.join(f'<span class="skill-tag">{skill}</span>' for skill in missing_skills)}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add skill recommendation visualization
                fig = go.Figure()
                for role, skills in insights['skill_gaps'].items():
                    fig.add_trace(go.Bar(
                        name=role,
                        x=skills,
                        y=[len(insights['skill_gaps'][role]) - i for i in range(len(skills))],
                        orientation='h'
                    ))
                fig.update_layout(
                    title="Recommended Skills by Role",
                    yaxis_title="Priority",
                    xaxis_title="Skills",
                    showlegend=True
                )
                st.plotly_chart(fig)
            else:
                st.success("Your skill set is well-rounded! Keep updating your skills to stay current.")

        # ATS Score and Optimization
        st.markdown("## 🎯 ATS Optimization")
        
        # Allow user to input target job description
        job_description = st.text_area(
            "Paste the job description you're targeting (optional)",
            help="This will help optimize your resume for ATS systems"
        )
        
        if st.button("Generate ATS-Optimized Resume"):
            with st.spinner("Optimizing your resume..."):
                # Generate optimized content
                optimized_content = self.optimizer.optimize_resume(resume_data, job_description)
                
                # Create before/after comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Original Skills")
                    self.display_skills(resume_data['skills'])
                
                with col2:
                    st.markdown("### Optimized Skills")
                    self.display_skills(optimized_content['skills'])
                
                # Generate and offer optimized PDF download
                pdf = self.optimizer.generate_pdf(optimized_content)
                pdf_output = pdf.output(dest='S').encode('latin1')
                
                st.download_button(
                    label="Download ATS-Optimized Resume",
                    data=pdf_output,
                    file_name="optimized_resume.pdf",
                    mime="application/pdf"
                )
        
        # Resume Tips
        st.markdown("## 💡 Pro Tips")
        tips = [
            {
                "category": "ATS Optimization",
                "tips": [
                    "Use standard section headers (Experience, Education, Skills)",
                    "Include keywords from the job description",
                    "Avoid fancy formatting and graphics",
                    "Use standard fonts (Arial, Calibri, Times New Roman)",
                    "Save your resume in PDF format"
                ]
            },
            {
                "category": "Content Structure",
                "tips": [
                    "Start bullet points with strong action verbs",
                    "Include quantifiable achievements",
                    "Keep sentences concise and clear",
                    "Maintain consistent formatting throughout",
                    "Proofread carefully for errors"
                ]
            },
            {
                "category": "Skills Presentation",
                "tips": [
                    "Group skills by category",
                    "Highlight skills mentioned in the job description",
                    "Include both technical and soft skills",
                    "Remove outdated or irrelevant skills",
                    "List skills in order of relevance"
                ]
            }
        ]
        
        for tip_category in tips:
            with st.expander(f"📌 {tip_category['category']} Tips"):
                for tip in tip_category['tips']:
                    st.markdown(f"""
                    <div class="tip-card">
                        <p>✨ {tip}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Industry Insights
        st.markdown("## 📈 Industry Insights")
        trending_skills = {
            "AI/ML": ["Python", "TensorFlow", "PyTorch", "NLP", "Computer Vision"],
            "Web Development": ["React", "Node.js", "TypeScript", "GraphQL", "Next.js"],
            "Cloud": ["AWS", "Azure", "Kubernetes", "Docker", "Terraform"],
            "Data": ["SQL", "Spark", "Snowflake", "Tableau", "Power BI"]
        }
        
        for category, skills in trending_skills.items():
            with st.expander(f"🚀 Trending in {category}"):
                for skill in skills:
                    relevance = random.randint(70, 100)
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>{skill}</h4>
                        <div class="skill-progress">
                            <div class="skill-progress-bar" style="width: {relevance}%"></div>
                        </div>
                        <p>Industry Relevance: {relevance}%</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Job Market Alignment
        if job_description:
            st.markdown("## 🎯 Job Market Alignment")
            alignment_score = self.calculate_job_alignment(resume_data, job_description)
            
            st.markdown(f"""
            <div class="insight-card">
                <h3>Job Alignment Score</h3>
                <div class="metric">
                    <div class="metric-value" style="color: {'#00c853' if alignment_score >= 80 else '#ffd600' if alignment_score >= 60 else '#ff3d00'}">
                        {alignment_score}%
                    </div>
                    <div class="metric-label">Match with Job Description</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Provide specific recommendations based on job description
            missing_keywords = self.identify_missing_keywords(resume_data, job_description)
            if missing_keywords:
                st.markdown("### 📝 Keyword Recommendations")
                st.markdown("Consider incorporating these keywords from the job description:")
                for keyword in missing_keywords:
                    st.markdown(f'<span class="skill-tag">{keyword}</span>', unsafe_allow_html=True)
    
    def calculate_job_alignment(self, resume_data, job_description):
        # Calculate similarity between resume and job description
        resume_text = " ".join([
            str(resume_data.get('text', '')),
            " ".join(str(skill) for skills in resume_data['skills'].values() for skill in skills),
            " ".join(str(exp) for exp in resume_data['experience'])
        ]).lower()
        
        job_description = job_description.lower()
        
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, resume_text, job_description).ratio()
        return int(similarity * 100)
    
    def identify_missing_keywords(self, resume_data, job_description):
        # Extract important keywords from job description
        doc = nlp(job_description)
        job_keywords = set(token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN'])
        
        # Get resume keywords
        resume_text = " ".join([
            str(resume_data.get('text', '')),
            " ".join(str(skill) for skills in resume_data['skills'].values() for skill in skills)
        ]).lower()
        resume_keywords = set(token.text.lower() for token in nlp(resume_text) if token.pos_ in ['NOUN', 'PROPN'])
        
        # Find missing important keywords
        return list(job_keywords - resume_keywords)

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


if __name__ == "__main__":
    analyzer = ModernResumeAnalyzer()
    analyzer.setup_page()
    analyzer.create_header()
    