from langchain.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat
import streamlit as st
import boto3

class LLMFactory:
    def __init__(self):
        # Setup bedrock (assume you have enabled anthropic.claude-v2 in Amazon Bedrock at us-east-1 region)
        self.bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    ) 

    def create_llm(self, llm_model):
        if llm_model == 'anthropic.claude-v2':
            inference_modifier = {'max_tokens_to_sample': 4096, 
                                  "temperature": 0.01,
                                  "top_k": 250,
                                  "top_p": 1,
                                  "stop_sequences": ["\n\nHuman"]}
            llm = Bedrock(model_id="anthropic.claude-v2",
                          client=self.bedrock_runtime, 
                          model_kwargs=inference_modifier)
        elif llm_model == 'anthropic.claude-3-sonnet':
            inference_modifier = {'max_tokens_to_sample': 4096, 
                                  "temperature": 0.01,
                                  "top_k": 250,
                                  "top_p": 1,
                                  "stop_sequences": ["\n\nHuman"]}
            llm = BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                          client=self.bedrock_runtime, 
                          model_kwargs=inference_modifier)
        else:
            st.error('LLM not supported yet')
            st.stop()
            llm = None
        return llm
