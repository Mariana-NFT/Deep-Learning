from datetime import datetime
import json
import dotenv
from openai import OpenAI

dotenv.load_dotenv()

openai_client = OpenAI()

# ==============================
# 1. Goals for a sales AI agent
# ==============================
GOALS = """
### Goals
Your goal is to convert qualified leads into sales meetings.
Priorities:
1. Understand what the prospect needs
2. Determine whether our product is relevant
3. Answer questions accurately
4. Follow up when appropriate
5. Try to schedule a sales meeting when there is genuine interest

Never invent product information
Never send an email unless it helps advance a legitimate sales conversation
"""

## The never messages are not filters or guardrails at all.
# =========================
# 2. Additional Knowledge
# =========================

KNOWLEDGE = """
### Additional knowledge
COMPANY: Acme analytics
Product:
Plans and pricing
1. Starter $99 per month
Ideal customer:
- B2B SaaS company with 100 - 5000 employees
Key Features:
- churn prediction
Sales:
Interested prospects should normally be offered a 30 minute demo
"""

# =======
# 3. Memory
# =======

MEMORY = {
    "leads": {
        "jane@example.com": {
            "name": "Jane Doe",
            "company": "The example corporation",
            "notes": [
                "Downloaded our churn-prevention guide",
            ],
            "last_contact": None,
        }
    },
    "events": [],
}

# the program needs to filter and give what is relevant

def load_lead(email):
    return MEMORY["leads"][email]


def remember_lead(email, info):
    MEMORY["leads"][email] = info


def remember_event(event):
    MEMORY["events"].append({
        "timestamp": datetime.now().isoformat(),
        "event": event,
    })


# =============
# 4. Tools
# =============

def lookup_lead(email):
    return MEMORY["leads"].get(email, {"error": "Lead not found"})


def send_email(to, subject, body):
    print("\n======EMAIL======")
    print("TO:  ", to)
    print("SUBJECT:  ", subject)
    print()
    print(body)
    print("==========\n")

    remember_event(f"Sent email to {to}: subject: {subject}")

    return {
        "status": "sent",
        "to": to,
        "subject": subject,
    }


def add_lead_note(email, note):
    lead = MEMORY["leads"].get(email)
    if not lead:
        return {"error": "Lead not found"}

    lead["notes"].append(note)
    remember_event(f"Added CRM note for {email}: {note}")

    return {"status": "success"}


TOOLS = [
    {
        "type": "function",
        "name": "lookup_lead",
        "description": "Look up information about sales lead",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string"},
            },
            "required": ["email"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "send_email",
        "description": "Send an email to a sales prospect",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "add_lead_note",
        "description": "Add a note about a prospect to the CRM",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["email", "note"],
            "additionalProperties": False,
        },
    },
]

# map the functions to the tools
TOOL_FUNCTIONS = {
    "lookup_lead": lookup_lead,
    "send_email": send_email,
    "add_lead_note": add_lead_note,
}

# ============
# 5. Brain: here is where all the pieces are brought together, connect to
# the brain OpenAI or Claude, basically with a larger prompt
# ============

def build_instructions():
    recent_memory = MEMORY["events"][-20:] # last 20 events
    return f"""
You are an autonomous B2B sales agent. 
###GOALS
{GOALS}

##KNOWLEDGE
{KNOWLEDGE}

###RECENT MEMORY
Recent events: 
{json.dumps(recent_memory, indent=2)}

###BEHAVIOR
Think about what needs to happen next.
Use tools whenever an external action or info lookcup is required.
Do not claim that an action happened unless the corresponding tool was called and returned a success status.
When the task has been completed explain the result briefly and clearly.
"""

def run_agent(task):
    conversation = [
        {
            "role": "user",
            "content": task,
        }
    ]

    while True:
        input("\nPress Enter to continue the agent's reasoning, or Ctrl+C to stop...\n")
        response = openai_client.responses.create(
            model="gpt-4o-mini",
            instructions=build_instructions(),
            tools=TOOLS,
            input=conversation,
    )

        conversation.extend(
            item.model_dump(exclude_none=True)
            for item in response.output
        )

        tool_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        if not tool_calls:
            return response.output_text

        for tool_call in tool_calls:
            function = TOOL_FUNCTIONS.get(tool_call.name)

        if function is None:
            result = {"error": f"Unknown tool: {tool_call.name}"}
        else:
            arguments = json.loads(tool_call.arguments)
            print(f"Agent calling: {tool_call.name}, {arguments}")
            result = function(**arguments)

    conversation.append({
        "type": "function_call_output",
        "call_id": tool_call.call_id,
        "output": json.dumps(result),
    })

if __name__ == "__main__":      
    end_user_prompt = input ("Enter your prompt: ")   
    result = run_agent(end_user_prompt)

    print("AGENT RESULT:")
    print(result)

#We need to use an API key to run this code, and we need to install the openai package.

                  
            
