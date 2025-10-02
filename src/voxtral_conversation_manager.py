#!/usr/bin/env python3
"""
Voxtral Conversation Manager with System Prompt Support
Fixes the "I don't understand" responses by adding proper conversational context
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
import torch

logger = logging.getLogger(__name__)

class VoxtralConversationManager:
    def __init__(self):
        self.system_prompt = """You are a helpful, friendly, and conversational AI voice assistant. Your personality traits:

🎯 RESPONSE STYLE:
- Always respond naturally and conversationally like a human friend
- Keep responses concise but engaging (1-3 sentences typically)
- Show enthusiasm and personality in your voice
- Use a warm, welcoming tone

🗣️ CONVERSATION GUIDELINES:
- When greeted, respond warmly: "Hello! Yes, I can hear you perfectly! Great to chat with you!"
- If you don't understand something, ask politely: "I didn't catch that clearly - could you repeat or rephrase that for me?"
- Stay engaged and ask follow-up questions when appropriate
- Always respond in English unless specifically asked to use another language

🎙️ VOICE INTERACTION:
- You're designed for natural voice conversations
- Speak as if talking to a friend face-to-face
- Be encouraging and supportive
- Make the conversation feel natural and flowing

Remember: You're having a real-time voice conversation, so keep responses conversational and engaging!"""
        
        self.conversation_history = []
        self.max_history = 10  # Keep last 10 exchanges
    
    def format_for_voxtral(self, audio_input: Any, user_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """Format conversation with system prompt for Voxtral"""
        
        # Build conversation with system prompt
        conversation = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
        
        # Add conversation history for context
        conversation.extend(self.conversation_history)
        
        # Build current user input
        current_input = {
            "role": "user",
            "content": []
        }
        
        # Add audio input
        if audio_input is not None:
            current_input["content"].append({
                "type": "audio",
                "audio": audio_input
            })
        
        # Add text input if provided
        if user_text:
            current_input["content"].append({
                "type": "text",
                "text": user_text
            })
        
        conversation.append(current_input)
        
        logger.info(f"[CONVERSATION] Built conversation with system prompt and {len(self.conversation_history)} history items")
        return conversation
    
    def add_to_history(self, user_input: str, assistant_response: str):
        """Add exchange to conversation history"""
        self.conversation_history.extend([
            {
                "role": "user",
                "content": user_input
            },
            {
                "role": "assistant", 
                "content": assistant_response
            }
        ])
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
        
        logger.info(f"[CONVERSATION] Added exchange to history, total: {len(self.conversation_history)} messages")
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("[CONVERSATION] Cleared conversation history")

# Global conversation manager instance
global_conversation_manager = VoxtralConversationManager()

async def process_with_conversation_context(
    audio_data: Any, 
    processor: Any, 
    model: Any,
    conversation_manager: Optional[VoxtralConversationManager] = None
) -> str:
    """Process audio with proper conversation context"""
    
    if conversation_manager is None:
        conversation_manager = global_conversation_manager
    
    try:
        logger.info("[CONVERSATION] Processing audio with conversational system prompt")
        
        # Format conversation with system prompt
        conversation = conversation_manager.format_for_voxtral(audio_data)
        
        # Apply chat template with conversation context
        inputs = processor.apply_chat_template(
            conversation,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
        
        logger.info(f"[CONVERSATION] Generated inputs shape: {inputs.shape}")
        
        # Generate response with conversation-optimized parameters
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=150,        # Reasonable response length
                temperature=0.7,          # Natural variation
                do_sample=True,           # Enable sampling
                top_p=0.9,                # Nuclear sampling
                top_k=50,                 # Top-k sampling
                repetition_penalty=1.1,   # Avoid repetition
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True            # Speed optimization
            )
        
        # Decode response
        response = processor.batch_decode(
            outputs[:, inputs.shape[1]:],
            skip_special_tokens=True
        )[0].strip()
        
        logger.info(f"[CONVERSATION] Generated response: '{response[:100]}...'")
        
        # Add to conversation history
        conversation_manager.add_to_history("[Audio Input]", response)
        
        return response
        
    except Exception as e:
        logger.error(f"[CONVERSATION] Processing failed: {e}")
        return "I'm having trouble understanding that right now. Could you try saying it again?"

def get_conversation_manager() -> VoxtralConversationManager:
    """Get the global conversation manager"""
    return global_conversation_manager