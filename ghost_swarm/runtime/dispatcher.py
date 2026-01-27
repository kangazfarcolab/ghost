"""
dispatcher.py - Ghost Swarm Orchestrator

Responsibilities:
1. Register/Unregister Agents
2. Route tasks to the correct agent
3. Manage agent lifecycle
"""

import time
import uuid

class Dispatcher:
    def __init__(self, memory_system):
        self.agents = {}  # {agent_id: agent_instance}
        self.memory = memory_system
        self.task_queue = []
    
    def register_agent(self, agent):
        """Add a new agent to the swarm."""
        agent_id = agent.name + "_" + str(uuid.uuid4())[:4]
        self.agents[agent_id] = agent
        agent.connect(self, self.memory)
        self.memory.log(f"Registered agent: {agent_id} ({agent.role})")
        return agent_id
    
    def dispatch(self, user_intent):
        """Route user intent to the best agent."""
        self.memory.log(f"User Intent: {user_intent}")
        
        # Simple keyword routing for now
        best_agent = None
        
        if "git" in user_intent.lower():
            target_role = "OpsAgent"
        elif "python" in user_intent.lower() or "def " in user_intent:
            target_role = "CoderAgent"
        else:
            target_role = "Generalist"
            
        # Find agent with role
        for aid, agent in self.agents.items():
            if agent.role == target_role:
                best_agent = agent
                break
        
        if best_agent:
            self.memory.log(f"Routing to {best_agent.name}...")
            response = best_agent.process(user_intent)
            return response
        else:
            return "No suitable agent found."

class BaseAgent:
    def __init__(self, name, role, model=None):
        self.name = name
        self.role = role
        self.model = model  # The Ghost v6 instance
        self.dispatcher = None
        self.memory = None
    
    def connect(self, dispatcher, memory):
        self.dispatcher = dispatcher
        self.memory = memory
    
    def process(self, input_text):
        """Default processing logic."""
        # Here we would use self.model(input_text)
        return f"[{self.name}] Processed: {input_text}"

