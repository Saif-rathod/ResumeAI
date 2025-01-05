import streamlit as st
import PyPDF2
import pickle
import string
import nltk
import spacy
import re
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load spaCy model
nlp = spacy.load('models/en_core_web_sm')

class ResumeAnalyzer:
    def __init__(self):
        self.skill_patterns = self.initialize_skill_patterns()

    def initialize_skill_patterns(self):
        return {
            'Programming Languages': [
                'python', 'java', 'javascript', 'c++', 'ruby', 'php', 'swift',
                'kotlin', 'go', 'rust', 'typescript', 'scala', 'r', 'matlab'
            ],
            'Web Technologies': [
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'django',
                'flask', 'spring', 'asp.net', 'jquery', 'bootstrap', 'sass'
            ],
            'Databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis',
                'elasticsearch', 'cassandra', 'dynamodb', 'firebase'
            ],
            'Cloud & DevOps': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
                'terraform', 'ansible', 'circleci', 'git', 'github', 'gitlab'
            ],
            'AI & Data Science': [
                'machine learning', 'deep learning', 'tensorflow', 'pytorch',
                'keras', 'scikit-learn', 'pandas', 'numpy', 'data analysis',
                'computer vision', 'nlp', 'ai'
            ],
            'Soft Skills': [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'project management', 'agile', 'scrum', 'critical thinking'
            ]
        }

    def extract_text_from_pdf(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata = reader.metadata
                text = "".join(page.extract_text() for page in reader.pages)
                return {'text': text.strip(), 'pages': len(reader.pages), 'metadata': metadata}
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = ' '.join(text.split())
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tokens = [token for token in text.split() if token not in stop_words]
        doc = nlp(' '.join(tokens))
        return ' '.join(token.lemma_ for token in doc)

    def extract_contact_info(self, text):
        contact_info = {'email': None, 'phone': None, 'linkedin': None}
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b(?:\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        contact_info['email'] = re.findall(email_pattern, text)[0] if re.findall(email_pattern, text) else None
        contact_info['phone'] = re.findall(phone_pattern, text)[0] if re.findall(phone_pattern, text) else None
        contact_info['linkedin'] = re.findall(linkedin_pattern, text)[0] if re.findall(linkedin_pattern, text) else None
        return contact_info

    def analyze_education(self, text):
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'school', 'education', 'graduated']
        return [sentence.strip() for sentence in nltk.sent_tokenize(text.lower()) if any(keyword in sentence for keyword in education_keywords)]

    def extract_experience(self, text):
        experience_keywords = ['experience', 'work', 'employment', 'job', 'career', 'position', 'role', 'company', 'organization']
        return [sentence.strip() for sentence in nltk.sent_tokenize(text) if any(keyword in sentence.lower() for keyword in experience_keywords)]

    def extract_skills(self, text):
        text = text.lower()
        skills_found = {category: [] for category in self.skill_patterns.keys()}
        for category, patterns in self.skill_patterns.items():
            skills_found[category] = [skill for skill in patterns if skill in text]
        return skills_found

    def calculate_detailed_ats_score(self, resume_text, job_description):
        scores = {}
        resume_skills = self.extract_skills(resume_text)
        job_skills = self.extract_skills(job_description)
        skill_scores = {category: len(set(resume_skills[category]).intersection(set(job_skills[category]))) / len(set(job_skills[category])) if job_skills[category] else 1.0 for category in self.skill_patterns.keys()}
        scores['skill_match'] = np.mean(list(skill_scores.values())) * 100
        processed_resume = self.preprocess_text(resume_text)
        processed_job = self.preprocess_text(job_description)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([processed_resume, processed_job])
        scores['content_similarity'] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
        word_count = len(resume_text.split())
        scores['length_score'] = min(100, (word_count / 500) * 100) if word_count < 500 else min(100, (1000 / word_count) * 100)
        weights = {'skill_match': 0.4, 'content_similarity': 0.4, 'length_score': 0.2}
        final_score = sum(scores[metric] * weights[metric] for metric in weights.keys())
        return {'final_score': round(final_score, 2), 'component_scores': {k: round(v, 2) for k, v in scores.items()}, 'skill_category_scores': {k: round(v * 100, 2) for k, v in skill_scores.items()}}

    def create_visualizations(self, analysis_results):
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15), facecolor='white')
        fig.suptitle('Resume Analysis Dashboard', fontsize=16, y=0.95)
        colors = {
            'primary': '#2ecc71',
            'secondary': '#3498db',
            'accent1': '#e74c3c',
            'accent2': '#f1c40f',
            'accent3': '#9b59b6',
            'accent4': '#e67e22'
        }
        ax1 = plt.subplot(2, 3, 1, projection='polar')
        score = analysis_results['ats_scores']['final_score']
        theta = np.linspace(0, 2*np.pi, 100)
        r = np.ones_like(theta)
        ax1.plot(theta, r, color='lightgray')
        ax1.fill_between(theta, 0, r, where=theta <= 2*np.pi*score/100, color=colors['primary'], alpha=0.5)
        ax1.set_title(f'ATS Score: {score}%', pad=20)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2 = plt.subplot(2, 3, 2)
        skill_scores = analysis_results['ats_scores']['skill_category_scores']
        categories = list(skill_scores.keys())
        scores = list(skill_scores.values())
        y_pos = np.arange(len(categories))
        ax2.barh(y_pos, scores, color=colors['secondary'])
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(categories)
        ax2.set_xlim(0, 100)
        ax2.set_title('Skill Category Match Scores')
        ax3 = plt.subplot(2, 3, 3)
        component_scores = analysis_results['ats_scores']['component_scores']
        ax3.pie(component_scores.values(), labels=[k.replace('_', ' ').title() for k in component_scores.keys()], autopct='%1.1f%%', colors=[colors['primary'], colors['accent1'], colors['accent2']])
        ax3.set_title('Score Components')
        ax4 = plt.subplot(2, 3, 4)
        all_skills = [skill for skills in analysis_results['skills'].values() for skill in skills]
        if all_skills:
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_skills))
                ax4.imshow(wordcloud)
            except Exception:
                ax4.text(0.5, 0.5, 'Skills Overview\n' + '\n'.join(all_skills), ha='center', va='center')
        ax4.axis('off')
        ax4.set_title('Skills Overview')
        ax5 = plt.subplot(2, 3, 5)
        experience = analysis_results['experience']
        if experience:
            y_pos = np.arange(len(experience))
            ax5.barh(y_pos, [1]*len(experience), color=colors['accent3'])
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels([exp[:30] + '...' if len(exp) > 30 else exp for exp in experience])
        ax5.set_title('Experience Timeline')
        ax6 = plt.subplot(2, 3, 6)
        education = analysis_results['education']
        if education:
            y_pos = np.arange(len(education))
            ax6.barh(y_pos, [1]*len(education), color=colors['accent4'])
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels([edu[:30] + '...' if len(edu) > 30 else edu for edu in education])
        ax6.set_title('Education Summary')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def generate_report(self, analysis_results):
        report = f"""
Resume Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. ATS Score Analysis
--------------------
Overall Score: {analysis_results['ats_scores']['final_score']}%

Component Scores:
- Skill Match: {analysis_results['ats_scores']['component_scores']['skill_match']}%
- Content Similarity: {analysis_results['ats_scores']['component_scores']['content_similarity']}%
- Length and Formatting: {analysis_results['ats_scores']['component_scores']['length_score']}%

2. Skills Analysis
-----------------
"""
        for category, skills in analysis_results['skills'].items():
            if skills:
                report += f"\n{category}:\n- {', '.join(skills)}"
        report += f"""

3. Contact Information
---------------------
"""
        for key, value in analysis_results['contact_info'].items():
            if value:
                report += f"{key.capitalize()}: {value}\n"
        report += f"""

4. Education
-----------
"""
        for edu in analysis_results['education']:
            report += f"- {edu}\n"
        report += f"""

5. Experience
------------
"""
        for exp in analysis_results['experience']:
            report += f"- {exp}\n"
        return report

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

        recommendations = []
        for path, details in career_paths.items():
            if any(keyword in all_skills or keyword in experience_text for keyword in details['keywords']):
                recommendations.append((path, details))

        return recommendations

    def analyze_resume(self, pdf_path, job_description):
        pdf_info = self.extract_text_from_pdf(pdf_path)
        resume_text = pdf_info['text']
        contact_info = self.extract_contact_info(resume_text)
        education = self.analyze_education(resume_text)
        experience = self.extract_experience(resume_text)
        skills = self.extract_skills(resume_text)
        ats_scores = self.calculate_detailed_ats_score(resume_text, job_description)
        analysis_results = {
            'contact_info': contact_info,
            'education': education,
            'experience': experience,
            'skills': skills,
            'ats_scores': ats_scores,
            'metadata': pdf_info['metadata'],
            'pages': pdf_info['pages']
        }
        visualizations = self.create_visualizations(analysis_results)
        report = self.generate_report(analysis_results)
        course_recommendations = self.recommend_courses(skills, experience)
        return {'analysis': analysis_results, 'visualizations': visualizations, 'report': report, 'course_recommendations': course_recommendations}

def main():
    st.title("Advanced Resume Analyzer")
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.header("Upload Resume")
    pdf_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    st.sidebar.header("Job Description")
    job_description = st.sidebar.text_area("Enter the job description")

    if st.sidebar.button("Analyze Resume"):
        if pdf_file and job_description:
            try:
                with open("temp_resume.pdf", "wb") as f:
                    f.write(pdf_file.getbuffer())

                analyzer = ResumeAnalyzer()
                st.write("Analyzing resume...")
                results = analyzer.analyze_resume("temp_resume.pdf", job_description)

                st.write("\n=== Resume Analysis Results ===")
                st.write("\nATS Scores:")
                st.write(f"Final Score: {results['analysis']['ats_scores']['final_score']}%")
                st.write("\nComponent Scores:")
                for component, score in results['analysis']['ats_scores']['component_scores'].items():
                    st.write(f"- {component.replace('_', ' ').title()}: {score}%")

                st.write("\nSkill Category Scores:")
                for category, score in results['analysis']['ats_scores']['skill_category_scores'].items():
                    st.write(f"- {category}: {score}%")

                st.write("\nContact Information:")
                for key, value in results['analysis']['contact_info'].items():
                    if value:
                        st.write(f"- {key.title()}: {value}")

                st.write("\nSkills Found:")
                for category, skills in results['analysis']['skills'].items():
                    if skills:
                        st.write(f"\n{category}:")
                        st.write(", ".join(skills))

                st.write("\nEducation:")
                for edu in results['analysis']['education']:
                    st.write(f"- {edu}")

                st.write("\nExperience:")
                for exp in results['analysis']['experience']:
                    st.write(f"- {exp}")

                st.write("\nVisualizations:")
                st.pyplot(results['visualizations'])

                st.write("\nSaving detailed report...")
                with open('resume_analysis_report.txt', 'w') as f:
                    f.write(results['report'])
                st.write("Detailed report saved as 'resume_analysis_report.txt'")

                st.write("\nRecommendations:")
                final_score = results['analysis']['ats_scores']['final_score']
                if final_score >= 80:
                    st.write("✅ Strong match for the position!")
                    st.write("- Consider highlighting specific project achievements")
                    st.write("- Prepare detailed examples of past experiences")
                elif final_score >= 60:
                    st.write("🟡 Good potential match with some gaps")
                    st.write("- Focus on strengthening missing skills")
                    st.write("- Add more specific examples of required technologies")
                else:
                    st.write("❌ Significant skill gaps identified")
                    st.write("- Consider additional training in required technologies")
                    st.write("- Revise resume to better align with job requirements")

                required_skills = set()
                for skills in analyzer.extract_skills(job_description).values():
                    required_skills.update(skills)

                current_skills = set()
                for skills in results['analysis']['skills'].values():
                    current_skills.update(skills)

                missing_skills = required_skills - current_skills
                if missing_skills:
                    st.write("\nMissing Skills:")
                    st.write(", ".join(missing_skills))

                st.write("\nCourse Recommendations:")
                for path, details in results['course_recommendations']:
                    st.write(f"{details['icon']} **{path}**:")
                    for course in details['courses']:
                        st.write(f"- [{course[0]}]({course[1]})")

                st.write("\nResume and Interview Preparation Videos:")
                st.write("Resume Preparation Videos:")
                for video in resume_videos:
                    st.write(f"- [{video}]({video})")
                st.write("Interview Preparation Videos:")
                for video in interview_videos:
                    st.write(f"- [{video}]({video})")

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
        else:
            st.error("Please upload a resume and enter a job description.")

if __name__ == "__main__":
    main()
