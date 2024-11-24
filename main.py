import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# Load environment variables
load_dotenv()

# Define agents and tools
def create_agents():
    instructions = """You are an agent designed to write and execute Python code to answer questions.
        You have access to a Python REPL, which you can use to execute python code.
        If you get an error, debug your code and try again.
        Only use the output of your code to answer the question.
        Return the Python code in a code block using triple backticks (```python) to make it easier to read.
        If it does not seem like you can write code to answer the question, just return \"I don't know\"  as the answer."""

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    # Python Agent
    tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )
    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    # CSV Agents for each CSV file
    csv_paths = ["VR.csv", "CHIP.csv", "GAMES.csv", "IMDB.csv", "TECH.csv"]
    csv_tools = []

    for path in csv_paths:
        csv_agent = create_csv_agent(
            llm=ChatOpenAI(temperature=0, model="gpt-4"),
            path=path,
            verbose=True,
            allow_dangerous_code=True,
        )
        csv_tools.append(
            Tool(
                name=f"CSV Agent for {path}",
                func=csv_agent.invoke,
                description=f"""Useful when you need to answer questions related to the CSV file: {path}.
                Takes as input the entire question and returns the answer after running pandas calculations.""",
            )
        )

    # Combine Python Agent and CSV Agents
    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor.invoke,
            description="""Useful when you need to transform natural language to Python and execute the code, returning the results of the code execution. DOES NOT ACCEPT CODE AS INPUT.""",
        ),
    ] + csv_tools

    # Create Grand Agent
    grand_agent = create_react_agent(
        prompt=base_prompt.partial(instructions=""),
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )
    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    return grand_agent_executor

# Streamlit GUI setup
st.title("Agent Assistant GUI")
st.header("Interact with Agents")
# st.text("Sometimes it will explode and show: An error occurred: list index out of range. It's fine, run it a few times, it only happens with complex-ish operations.")

# Initialize session state for chat history and code context
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "grand_agent_executor" not in st.session_state:
    st.session_state["grand_agent_executor"] = create_agents()

# Function to build context from chat history
def build_context(history):
    """Builds a conversation context from the chat history."""
    context = ""
    for user_query, agent_response in history:
        context += f"User: {user_query}\nAgent: {agent_response}\n"
    return context

# Job selection menu
st.subheader("Select a Python Agent Task")
python_tasks = [
    "Calculate the sum of squares of first 10 numbers",
    "Generate a list of prime numbers below 100",
    "Create a simple factorial calculator for the number 10"
]
selected_task = st.selectbox("Choose a task:", options=python_tasks)

if st.button("Execute Task"):
    with st.spinner("Processing the selected task..."):
        try:
            grand_agent_executor = st.session_state["grand_agent_executor"]

            # Define prompts for the tasks
            task_prompts = {
                "Calculate the sum of squares of first 10 numbers": "Generate a whole Python function to calculate the sum of squares of the first 10 numbers. Make sure to return the result explicitly.",
                "Generate a list of prime numbers below 100": "Generate a whole Python function to calculate all prime numbers below 100. Ensure the list of primes is returned explicitly. Return the Python code in a code block using triple backticks (```python) to make it easier to read.",
                "Create a simple factorial calculator for the number 10": "Generate a whole Python function to calculate the factorial of a given number. Return the result explicitly.",
            }

            # Get the selected task prompt
            task_prompt = task_prompts.get(selected_task)

            if not task_prompt:
                st.error("Invalid task selected.")
            else:
                # Pass the prompt to the agent and log the generated code
                response = grand_agent_executor.invoke({"input": task_prompt})
                print("Agent Full Response:", response)

                # Extract the output or log an error
                if isinstance(response, dict) and "output" in response:
                    generated_code = response["output"]
                    print("Generated Code:", generated_code)

                    # Extract the Python code block
                    if "```python" in generated_code:
                        start = generated_code.find("```python") + len("```python")
                        end = generated_code.find("```", start)
                        extracted_code = generated_code[start:end].strip()
                    else:
                        extracted_code = generated_code

                    # Automatically add result assignment if a function is found without usage
                    if "return" in extracted_code and "result =" not in extracted_code:
                        # Extract the function name and add a result assignment
                        function_name_start = extracted_code.find("def ") + len("def ")
                        function_name_end = extracted_code.find("(", function_name_start)
                        function_name = extracted_code[function_name_start:function_name_end].strip()

                        # Append result assignment for calling the function
                        extracted_code += f"\nresult = {function_name}(10)" if 'factorial' in function_name else f"\nresult = {function_name}()"

                    # Clear the execution context before running new code
                    exec_globals = {}

                    try:
                        # Execute the new code
                        exec(extracted_code, exec_globals)
                        result = exec_globals.get("result", None)

                        # Add result to chat history
                        if result is not None:
                            st.session_state["chat_history"].append((selected_task, f"Result: {result}"))
                            st.success(f"Task executed successfully! Result: {result}")
                        else:
                            error_message = "Execution successful, but no result was found. Ensure the code assigns the output to 'result'."
                            st.session_state["chat_history"].append((selected_task, error_message))
                            st.warning(error_message)
                    except Exception as e:
                        error_message = f"Error during execution: {str(e)}"
                        st.session_state["chat_history"].append((selected_task, error_message))
                        st.error(error_message)
                else:
                    error_message = "The agent did not generate a valid output."
                    st.session_state["chat_history"].append((selected_task, error_message))
                    st.error(error_message)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.session_state["chat_history"].append((selected_task, error_message))
            st.error(error_message)

# User input
st.subheader("Ask a Question About the CSVs or Generate a Program")
user_prompt = st.text_input("Enter your question or program request:", placeholder="Type your question here...")

if st.button("Execute Query"):
    if user_prompt:
        with st.spinner("Processing your query..."):
            try:
                grand_agent_executor = st.session_state["grand_agent_executor"]

                # Build context from history
                context = build_context(st.session_state["chat_history"])
                input_with_context = f"{context}User: {user_prompt}\nAgent:"

                # Invoke the agent with context
                response = grand_agent_executor.invoke({"input": input_with_context})

                # Extract the final answer from the response
                final_answer = response.get("output", "No output available.")

                # Update chat history
                st.session_state["chat_history"].append((user_prompt, final_answer))
                st.success("Query executed successfully!")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Display chat history dynamically
chat_placeholder = st.empty()

if st.session_state["chat_history"]:
    # Render the chat history dynamically using st.empty()
    with chat_placeholder.container():
        for idx, (user_query, agent_response) in enumerate(st.session_state["chat_history"]):
            message(user_query, is_user=True, key=f"user_{idx}")
            message(agent_response, key=f"agent_{idx}")
