import dspy

# Connect to your local Ollama model
lm = dspy.LM(
    model="ollama/gemma3:1b",
    api_base="http://localhost:11434",
    api_key="ollama"  # dummy key, required by DSPy but not used
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