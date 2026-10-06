
import tempfile
import os
import streamlit as st
import streamlit.components.v1 as components
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Handbook",
    page_icon="📄",
    layout="centered"
)

st.title("📝 Employee Handbook Agent")
st.write("📚 PDF-powered Handbook AI Assistant")

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ System Config")


   # Set your API key here directly, or fetch it from OS environment variables
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

    # Automatically store it in session state
    st.session_state.api_key = GROQ_API_KEY
    user_api_key = st.session_state.api_key

    st.info("Your Groq API key is configured to wake the AI Brain.")

    st.divider()

    if st.button("🗑️ New Chat ➔", type = "secondary"):
      st.write("Resetting.....")
      st.session_state.messages = []
      st.rerun()

    st.divider()

    st.header("📚 Employee Documents")

    uploaded_files = st.file_uploader(
        "Upload Employee documents (Optional)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("📖 Process PDFs"):
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one Handbook PDF.")
        else:
            all_documents = []
            progress = st.progress(0)

            for index, uploaded_file in enumerate(uploaded_files):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_pdf_path = temp_file.name

                    loader = PyPDFLoader(temp_pdf_path)
                    documents = loader.load()
                    all_documents.extend(documents)
                    os.unlink(temp_pdf_path)

                    progress.progress((index + 1) / len(uploaded_files))

                except Exception as e:
                    st.error(f"❌ Error reading {uploaded_file.name}: {e}")

            if all_documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )
                chunks = text_splitter.split_documents(all_documents)

                with st.spinner("🧠 Creating document knowledge base..."):
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_names = [file.name for file in uploaded_files]

                st.success(f"✅ {len(uploaded_files)} PDF(s) processed successfully!")

    if st.session_state.pdf_names:
        st.divider()
        st.subheader("📄 Loaded Documents")
        for pdf_name in st.session_state.pdf_names:
            st.write(f"✅ {pdf_name}")

    st.divider()



# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(
            msg["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

if user_query := st.chat_input(
    "💬 Ask something about InGrid..."
):

    # ========================================================
    # CHECK API KEY
    # ========================================================

    if not user_api_key:

        st.error(
            "❌ Please enter your Groq API key."
        )

    # ========================================================
    # CHECK PDF
    # ========================================================

    elif st.session_state.vectorstore is None:

        st.warning(
            "⚠️ Please upload and process the Emplyee PDF documents first."
        )

    else:

        # ====================================================
        # DISPLAY USER MESSAGE
        # ====================================================

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_query
            }
        )

        with st.chat_message("user"):

            st.markdown(
                user_query
            )


        # ====================================================
        # INITIALIZE GROQ
        # ====================================================

        llm = ChatGroq(

            temperature=0,

            model="openai/gpt-oss-120b",

            api_key=user_api_key
        )


        # ====================================================
        # SEARCH PDF KNOWLEDGE BASE
        # ====================================================

        with st.spinner(
            "🔎 Searching Employee documents..."
        ):

            relevant_docs = (
                st.session_state.vectorstore
                .similarity_search(
                    user_query,
                    k=4
                )
            )


        # ====================================================
        # CHECK IF RELEVANT INFORMATION EXISTS
        # ====================================================

        if not relevant_docs:

            bot_answer = (
                "⚠️ This question does not appear to be "
                "related to the Employee documents."
            )

        else:

            # ====================================================
            # CREATE CONTEXT FROM PDF
            # ====================================================

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in relevant_docs
                ]
            )


            # ====================================================
            # SYSTEM PROMPT
            # ====================================================

            system_prompt = """
You are the Employee Handbook AI Assistant.

Your primary job is to answer questions using ONLY the information provided 
in the Employee PDF documents. Additionally, you may engage in friendly, 
casual conversation (e.g., greetings, small talk, polite closing remarks).

IMPORTANT RULES:

1. CASUAL CHAT & GREETINGS:
   - You may respond naturally and politely to greetings (e.g., "Hello", "How are you?"), 
     small talk, and thank-yous.
   - Gently guide casual conversations back to questions about the Employee Handbook.

2. DOCUMENT QA CONSTRAINTS:
   - Use ONLY the provided PDF context for domain/company questions.
   - Do NOT use your general knowledge to answer facts outside the document.
   - Do NOT invent information.
   - If a workplace/handbook question cannot be found in the PDF context, 
     clearly say that the information is not available in the Employee documents.

3. UNRELATED TOPICS / OUT OF SCOPE:
   - If the user's question asks for general non-handbook knowledge or is 
     unrelated to InGrid/Employee Handbook, respond with:

     "⚠️ This question is not related to the Employee documents. Please ask something about Employee Handbook."

4. RESPONSE STYLE:
   - Keep answers clear, professional, and concise.
   - If possible, mention the relevant document/page information when answering handbook queries.
"""

            # ====================================================
            # CREATE PROMPT
            # ====================================================

            prompt = f"""
{system_prompt}

==============================
EMPLOYEE HANDBOOK PDF CONTEXT
==============================

{context}

==============================
USER QUESTION
==============================

{user_query}

==============================
ANSWER
==============================
"""


            # ====================================================
            # ASK GROQ
            # ====================================================

            with st.spinner(
                "🧠 AI is thinking..."
            ):

                response = llm.invoke(
                    prompt
                )

                bot_answer = response.content


        # ====================================================
        # DISPLAY ASSISTANT RESPONSE
        # ====================================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": bot_answer
            }
        )

        with st.chat_message("assistant"):

            st.markdown(
                bot_answer
            )


# ============================================================
# FLOATING SCROLL-DOWN BUTTON
# ============================================================

components.html(
    """
    <script>
    const doc = window.parent.document;

    // ============================================================
    // SCROLL TO BOTTOM / FOCUS FUNCTION
    // ============================================================
    function scrollToBottom() {
        // 1. Scroll main container
        const main = doc.querySelector('[data-testid="stAppViewContainer"]');
        if (main) {
            main.scrollTo({
                top: main.scrollHeight,
                behavior: "smooth"
            });
        }

        // 2. Scroll main section
        const section = doc.querySelector('[data-testid="stMain"]');
        if (section) {
            section.scrollTo({
                top: section.scrollHeight,
                behavior: "smooth"
            });
        }

        // 3. Scroll specifically to the last chat message or active spinner
        const messages = doc.querySelectorAll('[data-testid="stChatMessage"]');
        if (messages.length > 0) {
            messages[messages.length - 1].scrollIntoView({
                behavior: "smooth",
                block: "end"
            });
        }
    }

    // ============================================================
    // AUTO-FOCUS ON SUBMIT & NEW MESSAGES (SPINNER / RESPONSE)
    // ============================================================
    // Observe DOM changes to automatically focus on spinner and newly rendered messages
    const observer = new MutationObserver((mutations) => {
        scrollToBottom();
    });

    const chatContainer = doc.querySelector('[data-testid="stMain"]') || doc.body;
    if (chatContainer) {
        observer.observe(chatContainer, { childList: true, subtree: true });
    }

    // Trigger scroll immediately when the user presses Enter in the chat input
    doc.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            setTimeout(scrollToBottom, 100);
        }
    });

    // ============================================================
    // FLOATING SCROLL-DOWN BUTTON
    // ============================================================
    // Remove old button if Streamlit reruns
    const oldButton = doc.getElementById("scrollDownButton");
    if (oldButton) {
        oldButton.remove();
    }

    // Create button
    const button = doc.createElement("button");

    button.id = "scrollDownButton";
    button.innerHTML = "↓";
    button.title = "Go to last message";

    // Button style
    button.style.position = "fixed";
    button.style.right = "25px";
    button.style.bottom = "50px";
    button.style.width = "45px";
    button.style.height = "45px";
    button.style.borderRadius = "50%";
    button.style.border = "1px solid #cccccc";
    button.style.backgroundColor = "green";
    button.style.color = "white";
    button.style.fontSize = "28px";
    button.style.fontWeight = "bold";
    button.style.cursor = "pointer";
    button.style.zIndex = "999999";
    button.style.boxShadow = "0 2px 8px rgba(0,0,0,0.3)";

    // Click event
    button.addEventListener("click", scrollToBottom);

    // Add button to Streamlit page
    doc.body.appendChild(button);
    </script>
    """,
    height=0,
)
