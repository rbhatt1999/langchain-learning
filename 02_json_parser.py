"""
Exercise 2: Creating and Using a JSON Output Parser

Objective:
- Use LangChain's JsonOutputParser to force the LLM to return structured data.
- Define a schema using Pydantic.

Instructions:
1. Load GOOGLE_API_KEY.
2. Define a Pydantic model 'Movie' with fields: title, director, year, and genre.
3. Initialize the JsonOutputParser with the Movie model.
4. Create a PromptTemplate that:
   - Takes a 'movie_name' as input.
   - Includes the parser's formatting instructions.
5. Build a chain: Prompt | Model | Parser.
6. Test with at least 3 different movies and print the resulting Python dictionaries.
"""
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
load_dotenv()
# ---------------- MOVIE MODEL ----------------
class Movie(BaseModel):
    title: str = Field(description="The title of the movie")
    director: str = Field(description="The director of the movie")
    year: int = Field(description="The year the movie was released")
    genre: str = Field(description="The genre of the movie")
    rating: float = Field(description="The rating of the movie")
    revenue_millions: float = Field(description="The revenue of the movie in millions")

# ---------------- AI MODEL ----------------
ai = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

# ---------------- JSON OUTPUT PARSER ----------------
parser = JsonOutputParser(pydantic_object=Movie)
# ---------------- PROMPT TEMPLATE ----------------
prompt = PromptTemplate(
    template="Extract info about the movie {movie}.\n{format_instructions}",
    input_variables=["movie"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
# ---------------- CHAIN ----------------
chain = prompt | ai | parser
# ---------------- RESULT ----------------
result = chain.invoke({"movie": "Avenger end game"})
print(result)
