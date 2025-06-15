# app.py
import streamlit as st
from rag.retrieve import retrieve_context
import google.generativeai as genai
import chromadb
import re

# Setup Gemini Flash
genai.configure(api_key="AIzaSyDoGM_oH3vjWKPVFXVN0bZgYS-KPIfuxws")
model = genai.GenerativeModel("gemini-2.0-flash")

CHROMA_PATH = "./chroma_db"

def get_emotion_from_context(query: str) -> str:
    """Get the emotion from the most relevant context"""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection("empathetic_data")
        results = collection.query(query_texts=[query], n_results=1)
        
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            return results['metadatas'][0][0].get('emotion', 'neutral')
        return 'neutral'
    except Exception as e:
        st.error(f"Error getting emotion: {str(e)}")
        return 'neutral'

def generate_issue_from_context(emotion: str, user_query: str) -> str:
    """Generate a specific issue/condition based on emotion and user context"""
    try:
        issue_prompt = f"""Based on the user's emotional state and their message, identify a specific psychological or emotional issue/condition they might be experiencing.

Emotion detected: {emotion}
User message: {user_query}

Generate a concise, specific issue name (2-4 words maximum) that best describes what the user might be dealing with. 

Examples of good issue names:
- Chronic stress
- Social anxiety
- Work burnout
- Relationship conflict
- Academic pressure
- Sleep difficulties
- Low self-esteem
- Decision paralysis
- Imposter syndrome
- Financial anxiety
- Family tension
- Career uncertainty

Respond with ONLY the issue name, nothing else."""

        response = model.generate_content(issue_prompt)
        issue = response.text.strip()
        
        # Clean up the response and ensure it's concise
        issue = re.sub(r'^["\'\-\s]+|["\'\-\s]+$', '', issue)  # Fixed regex
        
        # If it's too long, truncate or use fallback
        if len(issue) > 25:
            issue = emotion.title() + " Issues"
            
        return issue.title()
        
    except Exception as e:
        st.error(f"Error generating issue: {str(e)}")
        return emotion.title() + " Related"

def generate_activities_for_emotion(emotion: str, user_query: str) -> list:
    """Generate specific activities for the detected emotion"""
    try:
        activity_prompt = f"""Generate 4-5 short, actionable activity suggestions for someone feeling "{emotion}". 
        
        User context: {user_query}
        
        Requirements:
        - Each activity should be 1-2 sentences maximum
        - Make them specific and actionable
        - Tailor them to the "{emotion}" emotional state
        - Focus on practical, immediate actions
        - Format as a simple list, one activity per line
        - No bullet points, numbers, or special formatting
        
        Example format:
        Take 5 deep breaths and focus on your exhale
        Write down three things you're grateful for today
        Go for a 10-minute walk outside
        Listen to calming music for 15 minutes
        Call a friend or family member
        """
        
        response = model.generate_content(activity_prompt)
        activities_text = response.text.strip()
        
        # Split into individual activities and clean them up
        activities = []
        for line in activities_text.split('\n'):
            line = line.strip()
            # Remove any bullet points, numbers, or formatting
            cleaned = re.sub(r'^[•\-*\d\.\s]+', '', line)
            if len(cleaned) > 10 and cleaned not in activities:
                activities.append(cleaned)
        
        return activities[:5]  # Return max 5 activities
        
    except Exception as e:
        st.error(f"Error generating activities: {str(e)}")
        return []

def separate_response_and_activities(response_text: str) -> str:  # Fixed return type
    """Separate the empathetic response from activity suggestions"""
    lines = response_text.split('\n')
    response_lines = []
    activity_started = False
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if we've reached the activities section
        if (any(keyword in line_lower for keyword in ['activities', 'suggestions', 'try these', 'consider', 'here are some']) and
            any(keyword in line_lower for keyword in ['help', 'try', 'do', 'practice'])):
            activity_started = True
            continue
            
        # If activities haven't started and line doesn't look like an activity, keep it
        if not activity_started:
            # Skip lines that look like activity lists
            if not (line.strip().startswith(('•', '-', '*')) or re.match(r'^\d+\.', line.strip())):
                response_lines.append(line)
    
    # Join the response lines back together
    clean_response = '\n'.join(response_lines).strip()
    
    return clean_response

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = 'neutral'
if 'current_issue' not in st.session_state:
    st.session_state.current_issue = 'General Wellness'
if 'current_activities' not in st.session_state:
    st.session_state.current_activities = []

# Page configuration
st.set_page_config(page_title="Empathetic Chatbot", layout="wide")
st.title("🤖 Empathetic Chatbot")

# Create layout with columns
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 Chat")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get emotion and generate response
        with st.spinner("Thinking..."):
            # Get emotion for the user query
            detected_emotion = get_emotion_from_context(prompt)
            st.session_state.current_emotion = detected_emotion
            
            # Generate specific issue based on emotion and query
            detected_issue = generate_issue_from_context(detected_emotion, prompt)
            st.session_state.current_issue = detected_issue
            
            # Get contexts for response generation
            contexts = retrieve_context(prompt)
            full_context = "\n\n".join(contexts)
            
            # Generate empathetic response (without activities)
            system_prompt = f"""You are an empathetic AI assistant. Use the following past dialogues to help generate a caring and emotionally aware response.

Context from similar conversations:
{full_context}

The detected emotion for this conversation is: {detected_emotion}

User: {prompt}

Please provide a thoughtful, empathetic response that acknowledges the user's emotional state of "{detected_emotion}". 

Focus ONLY on providing emotional support, validation, and understanding. Do NOT include any activity suggestions or recommendations in your response. Keep your response conversational and supportive."""

            try:
                # Generate empathetic response
                response = model.generate_content(system_prompt)
                bot_response = response.text
                
                # Clean the response to remove any activity suggestions that might have leaked through
                clean_response = separate_response_and_activities(bot_response)
                
                # Generate activities separately
                activities = generate_activities_for_emotion(detected_emotion, prompt)
                st.session_state.current_activities = activities
                
                # Add only the clean response to chat history
                st.session_state.messages.append({"role": "assistant", "content": clean_response})
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(clean_response)
                    
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

with col2:
    # Issue display - always at the top
    st.markdown("### 🎯 Current Issue")
    issue_container = st.container()
    with issue_container:
        st.markdown(f"**{st.session_state.current_issue}**")
    
    st.markdown("---")  # Visual separator
    
    # Suggested activities
    st.markdown("### 💡 Suggested Activities")
    
    activities_container = st.container()
    with activities_container:
        if st.session_state.current_activities:
            for i, activity in enumerate(st.session_state.current_activities, 1):
                st.markdown(f"**{i}.** {activity}")
        else:
            st.info("💭 Activities will appear here after you share your thoughts!")
    
    st.markdown("---")  # Visual separator
    
    # Display emotion history
    if st.session_state.messages:
        st.markdown("### 📊 Recent Issues")
        
        # Get issues from recent user messages
        user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
        recent_issues = []
        
        for i, msg in enumerate(user_messages[-5:], 1):  # Last 5 messages
            emotion = get_emotion_from_context(msg["content"])
            issue = generate_issue_from_context(emotion, msg["content"])
            recent_issues.append(f"{i}. {issue}")
        
        for issue in recent_issues:
            st.text(issue)
    
    st.markdown("---")  # Visual separator
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_emotion = 'neutral'
        st.session_state.current_issue = 'General Wellness'
        st.session_state.current_activities = []
        st.rerun()