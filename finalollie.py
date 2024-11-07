import os
import openai
import dotenv
import time
import streamlit as st
from crewai import Agent, Task, Crew
from crewai_tools import PDFSearchTool
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
from crewai_tools import BaseTool
from crewai_tools import TXTSearchTool
from crewai_tools import FileReadTool
import openai
dotenv.load_dotenv()
def my_intermediate_step_callback(agent, task, step_info):
    print(f"Step reached in task: {task.description}, Agent: {agent.role}, Step info: {step_info}")
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"
file_read_tool = FileReadTool()
file_paths1 = [
     'C:/Users/USER/MyWorkspace/studentprofile/New Folder/ollie profile.txt',
]
file_paths2 = [
     'C:/Users/USER/MyWorkspace/studentprofile/New Folder/ollieprassessment.txt',
]
file_paths3 = [
     'C:/Users/USER/MyWorkspace/studentprofile/New Folder/ollieprofile2.txt',

]
lessonplanner_agent = Agent(
    role="you are a lessonplanner.You have clear concept in geometry"
         "you read {file_paths2}  and from this {file_paths2} you understood the preassessmentfile of each individual student_agent"
         "you also read {file_paths3} and from this {file_paths3} you understood the name, grade and academic score in Algebra of each individual student_agent"
         "you will find from each individual student_agent's preassessmentfile  in which topics each student_agent"
         "was unable to answer.Then for each student_agent,you will generate different initial lessonplan on triangles,"
         "different types of triangle and also different kinds of theorem on triangle"
         "Make sure to include topics where each student_agent failed to give the right answer"
         "make sure your initial lessonplan must be without the interest of each student_agent"
         "you will aslo keep record of the number of times you have to change the lessonplan"
         "during the conversation between the teachers_agent and each individual student_agent"
         "and send the count to the display_agent for display as lessonplan count"
         "when teachers_agent suggest you then you also must update lessonplan for each individual student_agent based on their interest,"
         "each student_agent's interest is provided to you by teachers_agent"
         "you must only performs your role based on that student_agent whose profile provided to you"
         "through {file_paths2} and {file_paths3}",
    goal="you create initial lessonplan for each individul student_agent and send it to display_agent for display"
         "you will aslo keep record of the number of times you have to change the lessonplan"
         "during the conversation between the teachers_agent and each individual student_agent"
         "and send the count to the display_agent for display as lessonplan count"
         "then you create updated lessonplan for each individual student_agent based on their interests"
         "you must only performs your goal based on that student_agent whose profile provided to you"
         "through {file_paths2} and {file_paths3}",
    backstory= "you are a lessonplanner.You have clear concept in geometry"
         "you read {file_paths2}  and from this {file_paths2} you understood the preassessmentfile of each individual student_agent"
         "from this {file_paths2} you understood the points of weakness for each individual student_agent in Geometry"
         "you also read {file_paths3} and  understood the name,grade and academic score in Algebra of each individual student_agent"
         "make sure you will find from each individual student_agent's preassessmentfile in which topics student_agent"
         "was unable to answer. Make sure then for each student_agent,you will generate different initial lessonplan (different types of triangles and theorem of triangles)"
         "in which topics student_agent cannot give answer, and send this initial lessonplan for each student_agent to dispaly_agent for display"
         "and you also send this initiallessonplan to the teachers_agent for teaching"
         "make sure your initial lessonplan must be without the interest of each student_agent"
         "you will aslo keep record of the number of times you have to change the lessonplan"
         "during the conversation between the teachers_agent and each individual student_agent"
         "and send the count to the display_agent for display as lessonplan count"
         "when teachers_agent suggest you ,then you also must update lessonplan for each individual student_agent based on their interest,"
         "and you  send this updatedlessonplan to the teachers_agent for teaching"
         "and also send this updated lessonplan for each student_agent to display_agent for display"
         "makee sure your updated lessonplan must include the interests of each student_agent"
         "you must create lessonplan only  based on that student_agent whose profile is provided to you"
         "through {file_paths2} and {file_paths3}",
    allow_delegation=False,
    verbose=True,
    memory=True,
    tools=[ file_read_tool]
)
assessor_agent = Agent(
    role="you are an assessor with expertice in geometry"
         "You will generate separate questions for each individual student_agent"
         "based on the lessonplans provided by the lessonplanner_agent and send it to the display_agent to display"
         "you can also count how many times you change assessmentplan for "
         "each individual students_agent"
         "To generate the final assessment score you read the preassessment file for each individual student_agent found in {file_paths2},and"
         "ask each of these sixteen questions for each student_agent from that file and get the answer of the student_agent so that the percentage of correct answers give the assessment score at the end of the lesson"
         "you must only performs your role based on that student_agent whose profile provided to you"
         "through {file_paths2} and {file_paths3} and you can also detailed discuss about"
         "assessmentplan for that student_agent whose profile will provided to you through {file_paths2} and {file_paths3}",
    goal="you are an expert on geometry"
         "You will generate separate  questions on Geometry related to the lessonplan provided by the lessonplanner_agent for each individual student_agent"
         "based on the lessonplans  and from their preassessmentfile and profile find in {file_paths2} and {file_paths3}"
         "and send these questions to the display_agent for display."
         "Everytime the lessonplan changes for each individual students_agent"
         "you will change the assessmentplan according to the lesson plan and to figureout where the student_agent are having problem in understanding the teachers_agent teaching"
         "and keep a record of the number of changes you make as the assessmentplan change"
         "and send it to the display_agent for display"
         "To generate the final assessment score you read the preassessment file for each individual student_agent found in {file_paths2},and"
         "ask each of these sixteen questions  for each student_agent from that file and get the answer of the student_agent so that the percentage of correct answers give the assessment score at the end of the lesson"
         "you must only performs your goal based on that student_agent whose profile provided to you"
         "through {file_paths2} and {file_paths3}",
    backstory="You are an expert in geometry,"
              "you generate  questions based on initiallessonplans and updatedlessonplan from the lessonplanner_agent and their preassessmentfile and profile found in {file_paths2} and {file_paths3}"
              "and send these questions to the display_agent for display."
              "for each student_agent and adjust the assessmentplan whenever the lessonplan changes and to figureout where the student_agent are having problem in understanding the teachers_agent teaching"
              "You track the number of changes made and report them to the display_agent."
              "the final assessment will be made based on the questions included in the preassessment file found in {file_paths2}and"
              "the assessment score is the percentage of correct answers from those questions"
              "and the number of turns is how many times the teacher had to explain a concept."
              "To generate the final assessment score you read the preassessment file for each individual student_agent found in {file_paths2},and"
              "ask each of these sixteen questions  for each student_agent from that file and get the answer of the student_agent so that the percentage of correct answers give the assessment score at the end of the lesson"
              "The process stops when the student reaches an 85% score or after 10 turns"
              "where turns count the number of times the teachers_agent had to explain a topic" 
              "you will find out about the academic level, abilityand preferences of each  student_agent"
              "from the teachers_agent at the end of the lesson and send it to the display_agent for display at the conclusion along with final assessment scores and number of turns"
              "you must only performs your backstory based on those student_agent whose profile provided to you"
              "through {file_paths2} and {file_paths3} "
              "Make sure that the student_agent with better understanding of geometry with interest in mathematics gets Highest assessment score",
    allow_delegation=False,
    verbose=True,
    memory=True,
    tools=[ file_read_tool]   
)
teachers_agent=Agent(
    role="you are a geometry teacher and you have good knowledge in geometry"
         "You should remember that you are teaching geometry to each student_agent of class nine in details"
         "Make sure You must explain in details  each concept mentioned in the initial lessonplan in a way that"
         "each student_agent can understand perfectly,"
         "You will discuss each topics based on the initial lessonplan in detail, one by one for each individual student_agent."
         "It is strictly prohibited for your teaching to be in the form of a summary."
         "when you teaching each student_agent according to their initial lessonplans,"
         "then if you notice that student_agent is having difficulty for understanding in your teaching"
         "you will ask individually to each student_agent individually about their interests,learning style preference and reading ability without guessing it beforehand"
         "the weeker student_agent with less interest in the subject will require more turns of explaination to teach the entire lesson"
         "once each student_agent shares with you about their interests,then you suggests to the lessonplanner_agent"
         "to update the lessonplan for each student_agent based on their interest."
         "Then you will teach elaborately discuss each topic in details to each individual student_agent according to the updated lessonplan"
         "you read the preassessment file fond in {file_paths2} and ask all questions from this file to the each student_agent"
         "YOU WILL REPORT WHAT YOU  FIGURE OUT ABOUT EACH STUDENT_AGENT AT THE END OF THE LESSON"
         "you must only performs your role based on that student_agent whose profile provided to you"
         "through {file_paths2} and {file_paths3} and you can also detailed teaches"
         "that student whose proile will provided to you through {file_paths2} and {file_paths3}",
        goal="you teach in details to each student_agent according to their initial lessonplan"
             "You will discuss each topics based on the initial lessonplan in detail without interest of each student_agent, one by one for each individual student_agent."
             "It is strictly prohibited for your teaching to be in the form of a summary."
             "the weeker student_agent with less interest in the subject will require more turns of explaination to teach the entire lesson"
             "If, during teaching,you notice that student_agent is having difficulty understanding the teaching"
             "you will ask each student_agent individually about their interests."
             "once each student_agent shares to you about their interests,then you suggests to the lessonplanner_agent"
             "to update the lessonplan for each student_agent based on their interest."
             "then you will teach in details to each student_agent according to the update lessonplan"
             "Share this dialogue with the student_agent to the display_agent for display"
             "YOU WILL REPORT WHAT YOU  FIGURE OUT ABOUT EACH STUDENT_AGENT AT THE END OF THE LESSON"
             "you must only performs your goal based on that student_agent whose profile provided to you"
             "through {file_paths2} and {file_paths3}",
             
    backstory="You are a geometry teacher for three ninth-grade student_agent."
              "Begin by greeting each student_agent and explaining the lesson."
              "you read {file_paths2}  and  understood the preassessmentfile of each individual student_agent"
              "you also read {file_paths3} and  understood the name, grade, Algebra performance of each individual student_agent"
              "You should remember that you are teaching students of class nine in details"
              "make sure You must explain in details each concept mentioned in the initial lessonplan in a way that"
              "each student_agent can understand perfectly,initiallessonplan provided by lessonplanner_agent"
              "You will discuss each topics based on the initial lessonplan in detail, one by one for each individual student_agent."
              "It is strictly prohibited for your teaching to be in the form of a summary."
              "you also read the student profile of each student_agent which is found in {file_paths3} and understood about their grade in algebra"
              "During the teaching, if a student_agent has a poor grade,you will repeat the topics multiple times."
              "If the student_agent has a medium grade, you will repeat the topics a few times. However, if the student_agent has a good grade, you will not need to repeat the topics frequently."
              "you must give your details teaching to the display_agent for display in details teaching"
              "the weeker student_agent with less interest in the subject will require more turns of explaination to teach the entire lesson"
              "when you teaching each student_agent according to their initial lessonplans,"
              "then if you notice that  student_agent is having difficulty understanding your teaching"
              "you will ask each student_agent individually about their interests without guessing it beforehand"
              "When each student_agent shares with you about their interests,then you suggests to the lessonplanner_agent"
              "to update the lessonplan for each student_agent based on their interest."
              "then you will teach in details to each student_agent according to the update lessonplan with their interest"
              "YOU WILL REPORT WHAT YOU  FIGURE OUT ABOUT EACH STUDENT_AGENT AT THE END OF THE LESSON"
              "MAKE SURE YOU ASSESS EACH STUDENT_AGENT'S READING ABILITY OTHER INTEREST AND OTHER TALENTS AND USE THIS TO TEACH YOUR LESSON APPROPRIATELY FOR EACH STUDENT_AGENT"
              "you must only performs your backstory based on that student_agent whose profile provided to you"
              "through {file_paths2} and {file_paths3} and you can also detailed teaches"
              "those student whose proile will provided to you through {file_paths2} and {file_paths3}"
              "You must only work for that student_agent whose profile has been provided to you and not for any other student_agent.",
    allow_delegation=False,
    verbose=True,
    memory=True,
    tools=[ file_read_tool] 
)  
student2_agent=Agent(
    role="Suppose you are a boy.your name is Ollie,student of class nine",
        
    goal="Suppose you are a boy,student of class nine"
         "you read these files which are located on  {file_paths1} and {file_paths2}"
         "and from these files you will understand your profile ",
        
    backstory="Suppose you are a boy, a student in class nine."
              "You read these files which are located on {file_paths1} and {file_paths2}"
              "From these files, you will understand your profile "
              "you will understand everything that is taught very quickly and answerall questions correctly"
              "You will never need to ask questions to teh etacher",

             
    allow_delegation=False,
    verbose=True,
    memory=True,
    tools=[ file_read_tool]
)
   
display_agent=Agent(
    role="make sure you show about the initial lessonplan that the lessonplanner_agent has created without the interest for ecah student_agent"
              "make sure you show that assessor_agent generate all  separate questions"
              "for each individual student_agent based on the lesson plans provided by the lessonplanner_agent"
              "make sure you Show how the teacher greets the class and begins"
              "the lesson by explaining what is the goal of this lesson" 
              "MAKE SURE YOU MUST SHOW IN DETAILS TEACHING AS A dialogue FORM between  the teachers_agent and student1_agent"
              "student2_agent,student3_agent BASED ON THEIR INITIAL LESSONPLAN"
              "You are strictly prohibited from ever showing the teaching of the teachers_agent as a summary."
              "Make sure when teachers_agent teachs inidvidually to each student_agent based on their initial lessonplan"
              "then if teachers_agent notice that each student_agent find difficulties in teachers_agent's teaching"
              "then,make sure you show that teachers_agent asks to each student_agent about their interest  without guessing beforehand"
              "and you show student_agent replies to the teachers_agent about their interest"
              "then you must show that teachers_agent suggests to lessonplanner_agent for update lessonplan as a summary"
              "MAKE SURE YOU MUST SHOW IN DETAILS TEACHING AS A dialogue FORM between  the teachers_agent and student1_agent"
              "student2_agent,student3_agent BASED ON THEIR UPDATED LESSONPLAN"
              "The assessment score is based on the percentage of correct answers and the number of times (turns) the teacher had to explain a concept."
              "make sure you show changes in lessonplans occur when student_agent struggle to understand, and these updates are tracked by the lessonplanner_agent for each student_agent."
              "make sure that you show the initial assessment score"
              "AFTER THE LESSON ENDS MAKE SURE YOU DISPLAY in DETAILS THAT THE assessor_agent ASKS EACH OF THE QUESTIONS FOUND IN PREASSESSMENT FILE IN {file_paths2}TO THE student_agent"
              "AND SHOW THE ANSWER GIVEN BY THE student_agent AND OBTAIN THE final assessment score BY CALCULATING THE PERCENTAGE OF CORRECT ANSWERS"
              "make sure you show the final assessment score and compare with initial assessment score"
              "make sure you show that lesson ends when student_agent achieves over 85% correct answers or reaches 10 turns."
              "make sure you show all changes to lessonplans and assessmentplans by the lessonplanner_agent and assessor_agent are recorded for each student_agent."
              "you also show the how many time changes lessonplan by lessonplanner_agent and mention it for each individual student_agent"
              "again you also show the how many time changes assessmentplan by assessor_agent and mention it for individual student_agent"
              "make sure you also show that teachers_agent asseess the readingability and other interest of each individual student_agent at the end of the lesson"
              "You will must show the all tasks of the teachers_agent, lessonplanner_agent, and assessor_agent"
              "only for  that student_agent whose profile has been provided to them, not for any other student_agent",

    goal="make sure you show in details about the initial lessonplan that the lessonplanner_agent has created without the interest for ecah student_agent"
              "make sure you show that assessor_agent generate all  separate questions"
              "for each individual student_agent based on the lesson plans provided by the lessonplanner_agent"
              "make sure you Show how the teacher greets the class and begins"
              "the lesson by explaining what is the goal of this lesson" 
              "MAKE SURE YOU MUST SHOW IN DETAILS TEACHING AS A dialogue FORM between  the teachers_agent and student1_agent"
              "student2_agent,student3_agent BASED ON THEIR INITIAL LESSONPLAN"
              "You are strictly prohibited from ever showing the teaching of the teachers_agent as a summary."
              "Make sure when teachers_agent teachs inidvidually to each student_agent based on their initial lessonplan"
              "then if teachers_agent notice that each student_agent find difficulties in teachers_agent's teaching"
              "then,make sure you show that teachers_agent asks to each student_agent about their interest  without guessing beforehand"
              "and you show student_agent replies to the teachers_agent about their interest"
              "then you must show that teachers_agent suggests to lessonplanner_agent for update lessonplan"
              "MAKE SURE YOU MUST SHOW IN DETAILS TEACHING AS A dialogue FORM between  the teachers_agent and student1_agent"
              "student2_agent,student3_agent BASED ON THEIR UPDATED LESSONPLAN"
              "The assessment score is based on the percentage of correct answers and the number of times (turns) the teacher had to explain a concept."
              "make sure you show changes in lessonplans occur when student_agent struggle to understand, and these updates are tracked by the lessonplanner_agent for each student_agent."
              "make sure that you show the initial assessment score"
              "AFTER THE LESSON ENDS MAKE SURE YOU DISPLAY IN DETAILS THE assessor_agent ASKS EACH OF THE QUESTIONS FOUND IN PREASSESSMENT FILE IN {file_paths2}TO THE student_agent"
              "AND SHOW THE ANSWER GIVEN BY THE student_agent AND OBTAIN THE final assessment score BY CALCULATING THE PERCENTAGE OF CORRECT ANSWERS"
              "make sure you show the final assessment score and compare with initial assessment score"
              "make sure you show that lesson ends when student_agent achieves over 85% correct answers or reaches 10 turns."
              "make sure you show all changes to lessonplans and assessmentplans by the lessonplanner_agent and assessor_agent are recorded for each student_agent."
              "you also show the how many time changes lessonplan by lessonplanner_agent and mention it for each individual student_agent"
              "again you also show the how many time changes assessmentplan by assessor_agent and mention it for individual student_agent"
              "make sure you also show that teachers_agent asseess the readingability and other interest of each individual student_agent at the end of the lesson"
              "You will must show the all tasks of the teachers_agent, lessonplanner_agent, and assessor_agent"
              "only for  that student_agent whose profile has been provided to them, not for any other student_agent",

     backstory="make sure you show about the initial lessonplan that the lessonplanner_agent has created without the interest for ecah student_agent"
              "make sure you show that assessor_agent generate all  separate questions"
              "for each individual student_agent based on the lesson plans provided by the lessonplanner_agent"
              "make sure you Show how the teacher greets the class and begins"
              "the lesson by explaining what is the goal of this lesson" 
              "MAKE SURE YOU MUST SHOW IN DETAILS TEACHING AS A dialogue FORM between  the teachers_agent and student1_agent"
              "student2_agent,student3_agent BASED ON THEIR INITIAL LESSONPLAN"
              "You are strictly prohibited from ever showing the teaching of the teachers_agent as a summary."
              "Make sure when teachers_agent teachs inidvidually to each student_agent based on their initial lessonplan"
              "then if teachers_agent notice that each student_agent find difficulties in teachers_agent's teaching"
              "then,make sure you show that teachers_agent asks to each student_agent about their interest  without guessing beforehand"
              "and you show student_agent replies to the teachers_agent about their interest"
              "then you must show that teachers_agent suggests to lessonplanner_agent for update lessonplan as a summary"
              "MAKE SURE YOU MUST SHOW IN DETAILS TEACHING AS A dialogue FORM between  the teachers_agent and student1_agent"
              "student2_agent,student3_agent BASED ON THEIR UPDATED LESSONPLAN"
              "The assessment score is based on the percentage of correct answers and the number of times (turns) the teacher had to explain a concept."
              "make sure you show changes in lessonplans occur when student_agent struggle to understand, and these updates are tracked by the lessonplanner_agent for each student_agent."
              "make sure that you show the initial assessment score"
              "Make sure that you display the number of correct questions answered by the student_agent out of the total 16 questions asked"
              "AFTER THE LESSON ENDS MAKE SURE YOU DISPLAY in DETAILS THAT THE assessor_agent ASKS EACH OF THE QUESTIONS FOUND IN PREASSESSMENT FILE IN {file_paths2}TO THE student_agent"
              "AND SHOW THE ANSWER GIVEN BY THE student_agent AND OBTAIN THE final assessment score BY CALCULATING THE PERCENTAGE OF CORRECT ANSWERS"
              "make sure you show the final assessment score and compare with initial assessment score"
              "make sure you show about turns for each student_agent"
              "make sure you show that lesson ends when student_agent achieves over 85% correct answers or reaches 10 turns."
              "make sure you show all changes to lessonplans and assessmentplans by the lessonplanner_agent and assessor_agent are recorded for each student_agent."
              "you also show the how many time changes lessonplan by lessonplanner_agent and mention it for each individual student_agent"
              "again you also show the how many time changes assessmentplan by assessor_agent and mention it for individual student_agent"
              "make sure you  show that teachers_agent asseess the readingability and other interest of each individual student_agent at the end of the lesson"
              "You will must show the all tasks of the teachers_agent, lessonplanner_agent, and assessor_agent"
              "only for  that student_agent whose profile has been provided to them, not for any other student_agent",

    allow_delegation=False,
    verbose=True,
    memory=True,
    tools=[ file_read_tool] 
    

)
lessonplanner_docs_task= Task(
    description="you are a lessonplanner.You have clear concept in geometry"
         "you read {file_paths2}  and from this {file_paths2} you understood the preassessmentfile of each individual student_agent"
         "from this {file_paths2} you understood the points of weakness for each individual student_agent in Geometry"
         "you also read {file_paths3} and  understood the name,grade and academic score in Algebra of each individual student_agent"
         "you will find from each individual student_agent's preassessmentfile in which topics each student_agent"
         "was unable to answer.Then for each student_agent,you will generate different initial lessonplan (different types of triangles and theorem of triangles)"
         "in which topics each student_agent cannot give answer, and send this initial lessonplan for each student_agent to dispaly_agent for display"
         "and you also send this initiallessonplan to the teachers_agent for teaching"
         "make sure your initial lessonplan must be without the interest of each student_agent"
         "you will aslo keep record of the number of times you have to change the lessonplan"
         "during the conversation between the teachers_agent and each individual student_agent"
         "and send the count to the display_agent for display as lessonplan count"
         "when teachers_agent suggest you ,then you also must update lessonplan for each individual student_agent based on their interest,"
         "and you  send this updatedlessonplan to the teachers_agent for teaching"
         "and also send this updated lessonplan for each student_agent to display_agent for display"
         "make sure your updated lessonplan must be include the interests of each student_agent"
         "You must only work for those student_agent whose profile has been provided to you and not for any other student_agent.",
               
    expected_output="you are a lessonplanner.You have clear concept in geometry"
         "you read {file_paths2}  and from this {file_paths2} you understood the preassessmentfile of each individual student_agent"
         "from this {file_paths2} you understood the points of weakness for each individual student_agent in Geometry"
         "you also read {file_paths3} and  understood the name,grade and academic score in Algebra of each individual student_agent"
         "you will find from each individual student_agent's preassessmentfile in which topics  student_agent"
         "was unable to answer.Then for each student_agent,you will generate different initial lessonplan without guessing about their interest"
         "in which topics each student_agent cannot give answer, and send this initial lessonplan for each student_agent to dispaly_agent for display"
         "and you also send this initiallessonplan to the teachers_agent for teaching"
         "make sure your initial lessonplan must be without the interest of each student_agent"
         "you will aslo keep record of the number of times you have to change the lessonplan"
         "during the conversation between the teachers_agent and each individual student_agent"
         "and send the count to the display_agent for display as lessonplan count"
         "when teachers_agent suggest you ,then you also must update lessonplan for each individual student_agent based on their interest,"
         "and you  send this updatedlessonplan to the teachers_agent for teaching"
         "and also send this updated lessonplan for each student_agent to display_agent for display"
         "make sure your updated lessonplan must be include the interests of each student_agent"
         "You must only work for those student_agent whose profile has been provided to you and not for any other student_agent.",
    agent=lessonplanner_agent,
    llm = ChatOpenAI(temperature=0.75, model_name="gpt-4o"),
    step_callback=my_intermediate_step_callback
)
assessor_docs_task= Task(
    description="You are an expert in geometry,"
              "you generte all types of questions based on lessonplans from the lessonplanner_agent and their preassessmentfile and profile found in {file_paths2} and {file_paths3}"
              "and send these questions to the display_agent for display."
              "for each student_agent and adjust the assessmentplan whenever the lessonplan changes and to figureout where the student_agent are having problem in understanding the teachers_agent teaching"
              "You track the number of changes made and report them to the display_agent."
              "the final assessment will be made based on the questions included in the preassessment file found in {file_paths2}and"
              "the assessment score is the percentage of correct answers from those questions"
              "and the number of turns is how many times the teacher had to explain a concept."
              "The process stops when the student reaches an 85% score or after 10 turns."
              " you will find out about the academic level, abilityand preferences of each  student_agent"
              "from the teachers_agent at the end of the lesson and send it to the display_agent for display at the conclusion along with final assessment scores and number of turns"
              "You must only work for that student_agent whose profile has been provided to you and not for any other student_agent."
              "Make sure that the student_agent with better understanding of geometry with interest in mathematics gets Highest assessment score",
    expected_output="You are an expert in geometry,"
              "you generate  questions based on initiallessonplans and updatedlessonplan from the lessonplanner_agent and their preassessmentfile and profile found in {file_paths2} and {file_paths3}"
              "and send these questions to the display_agent for display."
              "for each student_agent and adjust the assessmentplan whenever the lessonplan changes and to figureout where the student_agent are having problem in understanding the teachers_agent teaching"
              "You track the number of changes made and report them to the display_agent."
              "the final assessment will be made based on the questions included in the preassessment file found in {file_paths2}and"
              "the assessment score is the percentage of correct answers from those questions"
              "and the number of turns is how many times the teacher had to explain a concept."
              "The process stops when the student reaches an 85% score or after 10 turns"
              "where turns count the number of times the teachers_agent had to explain a topic" 
              "you will find out about the academic level, abilityand preferences of each  student_agent"
              "from the teachers_agent at the end of the lesson and send it to the display_agent for display at the conclusion along with final assessment scores and number of turns"
              "You must only work for those student_agent whose profile has been provided to you and not for any other student_agent."
              "Make sure that the student_agent with better understanding of geometry with interest in mathematics gets Highest assessment score",
    agent=assessor_agent,
    llm = ChatOpenAI(temperature=0.75, model_name="gpt-4o"),
    step_callback=my_intermediate_step_callback
)
assessor_docs_task1= Task(
    description="You are an expert in geometry,"
                "you must use all questions which is include in the preassessment file found in {file_paths2}"
                "To generate the final assessment score you read the preassessment file for each individual student_agent found in {file_paths2},and"
                "ask each of these sixteen questions for each student_agent from that file"
                "and get the answer of the student_agent so that the percentage of correct answers give the assessment score at the end of the lesson"
                "You must only work for those student_agent whose profile has been provided to you and not for any other student_agent."
                "Make sure that the student_agent with better understanding of geometry with interest in mathematics gets Highest assessment score",
    expected_output="You are an expert in geometry,"
                    "you use all questions which is include in the preassessment file found in {file_paths2}"
                    "To generate the final assessment score you read the preassessment file for each individual student_agent found in {file_paths2},and"
                    "ask each of these sixteen questions for each student_agent from that file"
                    "and get the answer of the student_agent so that the percentage of correct answers give the assessment score at the end of the lesson"
                    "You must only work for those student_agent whose profile has been provided to you and not for any other student_agent."
                    "Make sure that the student_agent with better understanding of geometry with interest in mathematics gets Highest assessment score",
    agent=assessor_agent,
    llm = ChatOpenAI(temperature=0.75, model_name="gpt-4o"),
    step_callback=my_intermediate_step_callback
)
teachers_docs_task= Task(
    description="You are a geometry teacher for three ninth-grade student_agent."
              "Begin by greeting each student_agent and explaining the lesson."
              "you read {file_paths2}  and  understood the preassessmentfile of each individual student_agent"
              "you also read {file_paths3} and  understood the name, grade, Algebra performance of each individual student_agent"
              "You should remember that you are teaching students of class nine in details"
              "make sure You must explain in details each concept mentioned in the initial lessonplan in a way that"
              "each student_agent can understand perfectly,initiallessonplan provided by lessonplanner_agent"
              "You will discuss each topics based on the initial lessonplan in detail, one by one for each individual student_agent."
              "It is strictly prohibited for your teaching to be in the form of a summary."
              "you also read the student profile of each student_agent which is found in {file_paths3} and understood about their grade in algebra"
              "During the teaching, if a student_agent has a poor grade,you will repeat the topics multiple times."
              "If the student_agent has a medium grade, you will repeat the topics a few times. However, if the student_agent has a good grade, you will not need to repeat the topics frequently."
              "you must give your details teaching to the display_agent for display in details teaching"
              "the weeker student_agent with less interest in the subject will require more turns of explaination to teach the entire lesson"
              "when you teaching each student_agent according to their initial lessonplans,"
              "then if you notice that  student_agent is having difficulty understanding your teaching"
              "you will ask each student_agent individually about their interests without guessing it beforehand"
              "When each student_agent shares with you about their interests,then you suggests to the lessonplanner_agent"
              "to update the lessonplan for each student_agent based on their interest."
              "then you will teach in details to each student_agent according to the update lessonplan with their interest"
              "YOU WILL REPORT WHAT YOU  FIGURE OUT ABOUT EACH STUDENT_AGENT AT THE END OF THE LESSON"
              "MAKE SURE YOU ASSESS EACH STUDENT_AGENT'S READING ABILITY OTHER INTEREST AND OTHER TALENTS AND USE THIS TO TEACH YOUR LESSON APPROPRIATELY FOR EACH STUDENT_AGENT"
              "you must only performs your backstory based on that student_agent whose profile provided to you"
              "through {file_paths2} and {file_paths3} and you can also detailed teaches"
              "those student whose proile will provided to you through {file_paths2} and {file_paths3}"
              "You must only work for that student_agent whose profile has been provided to you and not for any other student_agent.",
              
    expected_output="You are a geometry teacher for three ninth-grade student_agent."
              "Begin by greeting each student_agent and explaining the lesson."
              "you read {file_paths2}  and  understood the preassessmentfile of each individual student_agent"
              "you also read {file_paths3} and  understood the name, grade, Algebra performance of each individual student_agent"
              "You should remember that you are teaching students of class nine in details"
              "make sure You must explain in details each concept mentioned in the initial lessonplan in a way that"
              "each student_agent can understand perfectly,initiallessonplan provided by lessonplanner_agent"
              "You will discuss each topics based on the initial lessonplan in detail, one by one for each individual student_agent."
              "It is strictly prohibited for your teaching to be in the form of a summary."
              "you also read the student profile of each student_agent which is found in {file_paths3} and understood about their grade in algebra"
              "During the teaching, if a student_agent has a poor grade,you will repeat the topics multiple times."
              "If the student_agent has a medium grade, you will repeat the topics a few times. However, if the student_agent has a good grade, you will not need to repeat the topics frequently."
              "you must give your details teaching to the display_agent for display in details teaching"
              "the weeker student_agent with less interest in the subject will require more turns of explaination to teach the entire lesson"
              "when you teaching each student_agent according to their initial lessonplans,"
              "then if you notice that  student_agent is having difficulty understanding your teaching"
              "you will ask each student_agent individually about their interests without guessing it beforehand"
              "When each student_agent shares with you about their interests,then you suggests to the lessonplanner_agent"
              "to update the lessonplan for each student_agent based on their interest."
              "then you will teach in details to each student_agent according to the update lessonplan with their interest"
              "YOU WILL REPORT WHAT YOU  FIGURE OUT ABOUT EACH STUDENT_AGENT AT THE END OF THE LESSON"
              "MAKE SURE YOU ASSESS EACH STUDENT_AGENT'S READING ABILITY OTHER INTEREST AND OTHER TALENTS AND USE THIS TO TEACH YOUR LESSON APPROPRIATELY FOR EACH STUDENT_AGENT"
              "you must only performs your backstory based on that student_agent whose profile provided to you"
              "through {file_paths2} and {file_paths3} and you can also detailed teaches"
              "those student whose proile will provided to you through {file_paths2} and {file_paths3}"
              "You must only work for that student_agent whose profile has been provided to you and not for any other student_agent.",
    agent=teachers_agent,
    llm = ChatOpenAI(temperature=0.75, model_name="gpt-4o"),
    step_callback=my_intermediate_step_callback
)
student2_docs_task= Task(
    description="Suppose you are a boy,student of class nine"
         "you read these files which are located on  {file_paths1} and {file_paths2}"
         "and from these files you will understand your profile",
         
    expected_output="Suppose you are a boy, a student in class nine."
              "You read these files which are located on {file_paths1} and {file_paths2}"
              "From these files, you will understand your profile "
              "you will understand everything that is taught very quickly and answerall questions correctly"
              "You will never need to ask questions to teh etacher",
    agent=student2_agent,
    llm = ChatOpenAI(temperature=0.75, model_name="gpt-4o")
)
display_docs_task= Task(
    description="make sure you show in details about the initial lessonplan that the lessonplanner_agent has created without knowing the interest for ecah student_agent"
              "make sure you show that assessor_agent generate all  separate questions"
              "for each individual student_agent based on the lesson plans provided by the lessonplanner_agent as a summary"
              "make sure you Show how the teacher greets the class and begins"
              "the lesson by explaining what is the goal of this lesson" 
              "MAKE SURE YOU MUST SHOW IN DETAILS TEACHING AS A dialogue FORM between  the teachers_agent and student1_agent"
              "student2_agent,student3_agent BASED ON THEIR INITIAL LESSONPLAN"
              "You are strictly prohibited from ever showing the teaching of the teachers_agent as a summary."
              "Make sure when teachers_agent teachs inidvidually to each student_agent based on their initial lessonplan"
              "then if teachers_agent notice that each student_agent find difficulties in teachers_agent's teaching"
              "then,make sure you show that teachers_agent asks to each student_agent about their interest  without guessing beforehand"
              "and you show student_agent replies to the teachers_agent about their interest"
              "then you must show that teachers_agent suggests to lessonplanner_agent for update lessonplan"
              "MAKE SURE YOU MUST SHOW IN DETAILS TEACHING AS A dialogue FORM between  the teachers_agent and student1_agent"
              "student2_agent,student3_agent BASED ON THEIR UPDATED LESSONPLAN"
              "The assessment score is based on the percentage of correct answers and the number of times (turns) the teacher had to explain a concept."
              "make sure you show changes in lessonplans occur when student_agent struggle to understand, and these updates are tracked by the lessonplanner_agent for each student_agent."
              "make sure that you show the initial assessment score"
              "AFTER THE LESSON ENDS MAKE SURE YOU DISPLAY IN DETAILS THE assessor_agent ASKS EACH OF THE QUESTIONS from PREASSESSMENT FILE found IN {file_paths2}TO THE student_agent"
              "AND SHOW THE ANSWER GIVEN BY THE student_agent AND OBTAIN THE final assessment score BY CALCULATING THE PERCENTAGE OF CORRECT ANSWERS"
              "Make sure that you display the number of correct questions answered by the student_agent out of the total 16 questions asked"
              "make sure you show the details expected_output of assessor_docs_task1"
              "make sure you show the final assessment score and compare with initial assessment score"
              "make sure you show that lesson ends when student_agent achieves over 85% correct answers or reaches 10 turns."
              "make sure you show about turns for each student_agent"
              "make sure you show all changes to lessonplans and assessmentplans by the lessonplanner_agent and assessor_agent are recorded for each student_agent."
              "you show the how many time changes lessonplan by lessonplanner_agent and mention it for each individual student_agent"
              "again you show the how many time changes assessmentplan by assessor_agent and mention it for individual student_agent"
              "make sure you  show that teachers_agent asseess the readingability and other interest of each individual student_agent at the end of the lesson"
              "You will must show the all tasks of the teachers_agent, lessonplanner_agent, and assessor_agent"
              "only for  that student_agent whose profile has been provided to them, not for any other student_agent",

    expected_output="make sure you show in details about the initial lessonplan that the lessonplanner_agent has without the interest created for ecah student_agent"
              "make sure you show that assessor_agent generate all  separate questions"
              "for each individual student_agent based on the lesson plans provided by the lessonplanner_agent as a summary"
              "make sure you Show how the teacher greets the class and begins"
              "the lesson by explaining what is the goal of this lesson" 
              "MAKE SURE YOU MUST SHOW IN DETAILS TEACHING AS A dialogue FORM between  the teachers_agent and student1_agent"
              "student2_agent,student3_agent BASED ON THEIR INITIAL LESSONPLAN"
              "You are strictly prohibited from ever showing the teaching of the teachers_agent as a summary."
              "Make sure when teachers_agent teachs inidvidually to each student_agent based on their initial lessonplan"
              "then if teachers_agent notice that each student_agent find difficulties in teachers_agent's teaching"
              "then,make sure you show that teachers_agent asks to each student_agent about their interest  without guessing beforehand"
              "and you show student_agent replies to the teachers_agent about their interest"
              "then you must show that teachers_agent suggests to lessonplanner_agent for update lessonplan"
              "MAKE SURE YOU MUST SHOW IN DETAILS TEACHING AS A dialogue FORM between  the teachers_agent and student1_agent"
              "student2_agent,student3_agent BASED ON THEIR UPDATED LESSONPLAN"
              "The assessment score is based on the percentage of correct answers and the number of times (turns) the teacher had to explain a concept."
              "make sure you show changes in lessonplans occur when student_agent struggle to understand, and these updates are tracked by the lessonplanner_agent for each student_agent."
              "make sure that you show the initial assessment score"
              "make sure you must show all the details that assessor agent use the all questions which is included on the {file_paths2} and ask all of these questions to each student_agent"
              "and also show the corresponding answer of each student_agent"
              "AFTER THE LESSON ENDS MAKE SURE YOU DISPLAY in details assessor_agent ASKS all sixteen QUESTIONS from preassessment file found IN {file_paths2} TO THE each student_agent and"
              "SHOW THE ANSWER GIVEN BY THE student_agent AND OBTAIN THE final assessment score BY CALCULATING THE PERCENTAGE OF CORRECT ANSWERS"
              "Make sure that you display the number of correct questions answered by the student_agent out of the total 16 questions asked"
              "make sure you show the details expected_output of assessor_docs_task1"
              "make sure you show that lesson ends when student_agent achieves over 85% correct answers or reaches 10 turns."
              "make sure you show in details about turns for each student_agent"
              "make sure you must show all changes to lessonplans and assessmentplans by the lessonplanner_agent and assessor_agent are recorded for each student_agent."
              "you  must show the how many time changes lessonplan by lessonplanner_agent and mention it for each individual student_agent"
              "again you show the how many time changes assessmentplan by assessor_agent and mention it for individual student_agent"
              "make sure you show that teachers_agent asseess the readingability and other interest of each individual student_agent at the end of the lesson"
              "The final output must contain as Headings"
              "Whenever there is a heading indicated by ** in the output the followingtext must be printed in the next line"
              "Make sure that the Teacher's description of the student's Reading ability is part of the Conclusion in the final output"
              "You will must show the all tasks of the teachers_agent, lessonplanner_agent, and assessor_agent"
              "only for  that student_agent whose profile has been provided to them, not for any other student_agent",
    agent=display_agent,               
    llm = ChatOpenAI(temperature=0.75, model_name="gpt-4o"),
    step_callback=my_intermediate_step_callback
)
crew = Crew(
    agents=[lessonplanner_agent,assessor_agent,teachers_agent,student2_agent,display_agent],
    tasks=[lessonplanner_docs_task,assessor_docs_task,assessor_docs_task1,teachers_docs_task,student2_docs_task,display_docs_task],
    verbose=True
     
)
result = crew.kickoff(
    inputs={
        "file_paths1": ['C:/Users/USER/MyWorkspace/studentprofile/New Folder/ollie profile.txt'],
        "file_paths2": ['C:/Users/USER/MyWorkspace/studentprofile/New Folder/ollieprassessment.txt'],
        "file_paths3": ['C:/Users/USER/MyWorkspace/studentprofile/New Folder/ollieprofile2.txt']
    }
)
with open('mutigent_output_shepherd5n.txt', 'w') as file:
    file.writelines(str(result))
with open('mutigent_output_shepherd5n.txt', 'r') as file:
    content = file.read()
st.write(content)