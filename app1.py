# app.py
import streamlit as st
from rag.retrieve import retrieve_context
import google.generativeai as genai
import chromadb
import re
from datetime import datetime
from typing import List, Dict, Tuple

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

def generate_issue_from_context(emotion: str, user_query: str, conversation_history: str = "") -> str:
    """Generate a specific issue/condition based on emotion, user context, and conversation history"""
    try:
        issue_prompt = f"""Based on the user's emotional state, their current message, and the conversation history, identify a specific psychological or emotional issue/condition they might be experiencing.

Emotion detected: {emotion}
Current user message: {user_query}

Conversation history:
{conversation_history}

Generate a concise, specific issue name (2-4 words maximum) that best describes what the user might be dealing with, considering the full context of the conversation.

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
        issue = re.sub(r'^["\'\-\s]+|["\'\-\s]+$', '', issue)
        
        # If it's too long, truncate or use fallback
        if len(issue) > 25:
            issue = emotion.title() + " Issues"
            
        return issue.title()
        
    except Exception as e:
        st.error(f"Error generating issue: {str(e)}")
        return emotion.title() + " Related"

def generate_activities_for_emotion(emotion: str, user_query: str, conversation_context: str = "") -> list:
    """Generate specific activities for the detected emotion with conversation context"""
    try:
        activity_prompt = f"""Generate 4-5 short, actionable activity suggestions for someone feeling "{emotion}". 
        
        Current user message: {user_query}
        Conversation context: {conversation_context}
        
        Requirements:
        - Each activity should be 1-2 sentences maximum
        - Make them specific and actionable
        - Tailor them to the "{emotion}" emotional state
        - Consider the conversation context for personalized suggestions
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

def get_conversation_history(messages: List[Dict], max_messages: int = 10) -> str:
    """Get formatted conversation history for context"""
    if not messages:
        return ""
    
    # Get the last max_messages messages
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    history_parts = []
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        history_parts.append(f"{role}: {content}")
    
    return "\n".join(history_parts)

def get_conversation_summary(messages: List[Dict]) -> str:
    """Generate a summary of the conversation for better context understanding"""
    if len(messages) < 4:  # Not enough messages to summarize
        return ""
    
    try:
        # Get conversation history
        history = get_conversation_history(messages, max_messages=8)
        
        summary_prompt = f"""Analyze this conversation and provide a brief summary of the main themes, emotions, and issues discussed:

Conversation:
{history}

Provide a concise summary (2-3 sentences) that captures:
1. The main emotional themes
2. Key issues or concerns mentioned
3. Any progress or patterns in the conversation

Summary:"""

        response = model.generate_content(summary_prompt)
        return response.text.strip()
        
    except Exception as e:
        st.error(f"Error generating conversation summary: {str(e)}")
        return ""

def separate_response_and_activities(response_text: str) -> str:
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

def store_conversation_context(user_message: str, bot_response: str, emotion: str, issue: str):
    """Store conversation context for future retrieval"""
    try:
        if 'conversation_contexts' not in st.session_state:
            st.session_state.conversation_contexts = []
        
        context_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'bot_response': bot_response,
            'emotion': emotion,
            'issue': issue
        }
        
        st.session_state.conversation_contexts.append(context_entry)
        
        # Keep only last 20 context entries to manage memory
        if len(st.session_state.conversation_contexts) > 20:
            st.session_state.conversation_contexts = st.session_state.conversation_contexts[-20:]
            
    except Exception as e:
        st.error(f"Error storing conversation context: {str(e)}")

def get_relevant_conversation_context(current_query: str, emotion: str) -> str:
    """Retrieve relevant conversation context based on current query and emotion"""
    if 'conversation_contexts' not in st.session_state or not st.session_state.conversation_contexts:
        return ""
    
    try:
        # Simple relevance matching - in a production system, you might use embeddings
        relevant_contexts = []
        
        for context in st.session_state.conversation_contexts[-10:]:  # Last 10 contexts
            # Check if emotions match or if there are common keywords
            if (context['emotion'] == emotion or 
                any(word in context['user_message'].lower() for word in current_query.lower().split() if len(word) > 3)):
                relevant_contexts.append(f"Previous: {context['user_message'][:100]}... -> {context['bot_response'][:100]}...")
        
        return "\n".join(relevant_contexts[-3:]) if relevant_contexts else ""  # Return last 3 relevant contexts
        
    except Exception as e:
        st.error(f"Error retrieving conversation context: {str(e)}")
        return ""

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = 'neutral'
if 'current_issue' not in st.session_state:
    st.session_state.current_issue = 'General Wellness'
if 'current_activities' not in st.session_state:
    st.session_state.current_activities = []
if 'conversation_contexts' not in st.session_state:
    st.session_state.conversation_contexts = []
if 'conversation_summary' not in st.session_state:
    st.session_state.conversation_summary = ""

# Page configuration
st.set_page_config(page_title="Empathetic Chatbot", layout="wide")
st.title("Healing starts with talking... ")

# Create layout with columns
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 Chat")
    
    # Create a container for the chat messages with fixed height
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages in a scrollable container
        if st.session_state.messages:
            # Create a container with fixed height for scrolling
            messages_placeholder = st.empty()
            
            with messages_placeholder.container():
                # Use st.container with height parameter for scrolling
                with st.container(height=500):
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
        else:
            # Show empty state
            with st.container(height=500):
                st.info("👋 Hi there! How are you feeling today? I'm here to listen and support you.")
    
    # Fixed input at the bottom - this stays in place
    st.markdown("---")  # Visual separator
    
    # Chat input - this will stay at the bottom
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get emotion and generate response
        with st.spinner("Thinking..."):
            # Get emotion for the user query
            detected_emotion = get_emotion_from_context(prompt)
            st.session_state.current_emotion = detected_emotion
            
            # Get conversation history and summary
            conversation_history = get_conversation_history(st.session_state.messages[:-1])  # Exclude current message
            conversation_summary = get_conversation_summary(st.session_state.messages[:-1])
            
            # Update conversation summary in session state
            if conversation_summary:
                st.session_state.conversation_summary = conversation_summary
            
            # Generate specific issue based on emotion, query, and history
            detected_issue = generate_issue_from_context(detected_emotion, prompt, conversation_history)
            st.session_state.current_issue = detected_issue
            
            # Get contexts for response generation (from RAG)
            contexts = retrieve_context(prompt)
            full_context = "\n\n".join(contexts)
            
            # Get relevant conversation context
            relevant_context = get_relevant_conversation_context(prompt, detected_emotion)
            
            # Generate empathetic response with full context
            system_prompt = f"""You are an empathetic AI assistant. Use the following information to generate a caring, emotionally aware, and contextually relevant response.

RAG Context from similar conversations:
{full_context}

Conversation History:
{conversation_history}

Conversation Summary:
{st.session_state.conversation_summary}

Relevant Previous Context:
{relevant_context}

The detected emotion for this conversation is: {detected_emotion}

Current User Message: {prompt}

Instructions:
- Acknowledge the conversation history and show that you remember what was discussed before
- Provide a thoughtful, empathetic response that considers the user's emotional state of "{detected_emotion}"
- Reference previous parts of the conversation when relevant
- Focus ONLY on providing emotional support, validation, and understanding
- Do NOT include any activity suggestions or recommendations in your response
- Keep your response conversational, supportive, and contextually aware
- If this is a follow-up to previous issues, acknowledge the continuity"""

            try:
                # Generate empathetic response
                response = model.generate_content(system_prompt)
                bot_response = response.text
                
                # Clean the response to remove any activity suggestions that might have leaked through
                clean_response = separate_response_and_activities(bot_response)
                
                # Generate activities with conversation context
                conversation_context = f"{conversation_history}\n\nCurrent issue: {detected_issue}"
                activities = generate_activities_for_emotion(detected_emotion, prompt, conversation_context)
                st.session_state.current_activities = activities
                
                # Store conversation context for future use
                store_conversation_context(prompt, clean_response, detected_emotion, detected_issue)
                
                # Add only the clean response to chat history
                st.session_state.messages.append({"role": "assistant", "content": clean_response})
                
                # Rerun to update the display
                st.rerun()
                    
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

with col2:
    # Issue display - first position
    st.markdown("### Current Issue")
    issue_container = st.container()
    with issue_container:
        st.markdown(f"**{st.session_state.current_issue}**")
    
    st.markdown("---")  # Visual separator
    
    # Suggested activities - second position
    st.markdown("### Suggested Activities")
    
    activities_container = st.container()
    with activities_container:
        if st.session_state.current_activities:
            for i, activity in enumerate(st.session_state.current_activities, 1):
                st.markdown(f"**{i}.** {activity}")
        else:
            st.info("💭 Activities will appear here after you share your thoughts!")
    
    st.markdown("---")  # Visual separator
    
    # Conversation summary - third position
    if st.session_state.conversation_summary and len(st.session_state.messages) > 4:
        st.markdown("### 📝 Conversation Summary")
        summary_container = st.container()
        with summary_container:
            st.markdown(f"*{st.session_state.conversation_summary}*")
        st.markdown("---")
    
    # Display recent issues from conversation history
    if st.session_state.messages:
        st.markdown("### Recent Issues")
        
        # Get issues from conversation contexts if available, otherwise generate from recent messages
        if 'conversation_contexts' in st.session_state and st.session_state.conversation_contexts:
            recent_contexts = st.session_state.conversation_contexts[-5:]
            for i, context in enumerate(recent_contexts, 1):
                st.text(f"{i}. {context['issue']}")
        else:
            # Fallback to old method
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
    if st.button("Clear Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_emotion = 'neutral'
        st.session_state.current_issue = 'General Wellness'
        st.session_state.current_activities = []
        st.session_state.conversation_contexts = []
        st.session_state.conversation_summary = ""
        conversation_history = ""
        st.rerun()