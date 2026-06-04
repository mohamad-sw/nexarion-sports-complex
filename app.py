import os
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.environ["GROQ_API_KEY"]
)

dspy.configure(lm=lm)


class DoubleChainOfThoughModule(dspy.Module):
   def __init__(self):
      self.cot1 = dspy.ChainOfThought('question -> step_by_step_thought')
      self.cot2 = dspy.ChainOfThought('question, thought -> one_word_answer')

   def forward(self, question):
      thought = self.cot1(question=question).step_by_step_thought
      answer = self.cot2(question=question, thought=thought )
      return dspy.Prediction(question=question, answer=answer)
   
      

double_cot = DoubleChainOfThoughModule()



prediction = double_cot(question="what is the birth date of the first man who landed on the moon")
print(prediction)
print(lm.inspect_history(2))