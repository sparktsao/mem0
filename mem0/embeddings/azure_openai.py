from openai import OpenAI

from mem0.embeddings.base import EmbeddingBase


class AzureOpenAIEmbedding(EmbeddingBase):
    def __init__(self, model="text-embedding-ada-002"):
        
        import os
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_key = os.getenv("AZURE_API_KEY"),  
            api_version = os.getenv("AZURE_API_VERSION"),
            azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.environ["DEPLOYMENT_NAME"]
        )
        self.model = model
        self.dims = 1536
        
    def embed(self, text):
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.

        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        ## SPARKTSAO
        print(self.model, text)
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )
