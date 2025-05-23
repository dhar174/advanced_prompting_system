from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import difflib

from Levenshtein import ratio
import openai
import os
import re
import time
import random


from tqdm import tqdm
from conversation_manager import output_type_determination, OutputType
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Tuple, Optional
import tiktoken

import complexity_measures
from complexity_measures import (
    Plan,
    PlanStep,
    Subtask,
    cosine_similarity_custom,
    get_embedding,
)


# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

META_PROMPT_TEMPLATE = """
You are an expert prompt engineer tasked with optimizing the prompts used for solving complex tasks.

Given the following performance data from previous interactions:

{performance_data}

Analyze the strengths and weaknesses of the current prompting strategy. Provide detailed suggestions for improving the prompt to enhance solution quality, creativity, and reliability.

Your suggestions should include modifications to:
- Instructions clarity
- Response structure
- Role assignments
- Any other relevant aspects

Provide your recommendations in a structured JSON format as follows:

{{
    "prompt_modifications": [
        "Suggestion 1",
        "Suggestion 2",
        ...
    ]
}}
"""

example_interaction = {
    "steps": [
        {"description": "Understand the problem statement", "remaining_budget": 4},
        {"description": "Identify the key components", "remaining_budget": 3},
        {"description": "Formulate a plan to solve the problem", "remaining_budget": 2},
        {"description": "Execute the plan step-by-step", "remaining_budget": 1},
        {
            "description": "Review the solution and verify its correctness",
            "remaining_budget": 0,
        },
    ],
    "reflections": [
        {
            "content": "This step correctly identifies the product rule needed for differentiation",
            "reward": 1.0,
        },
        {"content": "Variables u(x) and v(x) are correctly assigned", "reward": 1.0},
        {
            "content": "Derivatives u'(x) and v'(x) are computed correctly",
            "reward": 1.0,
        },
        {
            "content": "Incorrect application of the quotient rule instead of the product rule",
            "reward": 0.4,
        },
        {
            "content": "Realizing the mistake in applying the quotient rule instead of the product rule. I need to correct this approach",
            "reward": 0.0,
        },
        {
            "content": "The product rule is correctly applied to obtain the final derivative",
            "reward": 1.0,
        },
    ],
    "answer": "The derivative of f(x) = sin(x) * e^x is f'(x) = cos(x) * e^x + sin(x) * e^x",
    "final_reward": 0.85,
}

examples_for_prompt_instructions = [
    f"""
<count>6</count>
<thinking>To solve 'Plan the best strategy for a chess endgame', I will analyze the given position and consider various tactical and strategic options. The position is as follows: White has a King on g1, Rook on c1, Pawns on a2, b3, e3, g2, h2; Black has a King on g8, Rook on f8, Pawns on a7, b6, e6, g7, h7. It's White to move.</thinking>
<step>Assess the current position and identify key features.</step>
<count>5</count>
<thinking>The key features are: material is equal with both sides having a rook and five pawns. White's rook is on an open file, while Black's rook is not actively placed. The pawn structures are symmetrical on the kingside but differ on the queenside. White has potential to penetrate on the c-file.</thinking>
<step>Identify weaknesses in Black's position and potential targets for infiltration.</step>
<count>4</count>
<thinking>Considering the open c-file and the possibility of activating the rook, I will evaluate moves like Rc1-c7 to pressure Black's seventh rank. I must also be cautious of any counterplay Black might have.</thinking>
<step>Calculate the viability of Rc1-c7 and anticipate Black's responses.</step>
<count>3</count>
<thinking>If I play Rc7, Black might respond with Rf8-d8 to contest the file. Alternatively, Black could play e6-e5 to activate their king and create pawn breaks. I need to prepare for these possibilities by supporting my rook's advancement and preparing my king's activation.</thinking>
<step>Develop a plan that includes Rc7 while preparing for Black's counterplay.</step>
<count>2</count>
<thinking>I can proceed with Rc7, and if Black contests the file, I can exchange rooks and activate my king with Kg1-f2. This would lead to a favorable king and pawn endgame due to my better pawn structure on the queenside. Additionally, advancing my pawns on the queenside could create passed pawn opportunities.</thinking>
<step>Execute Rc7 and prepare to transition into a favorable endgame through exchanges and king activity.</step>
<count>1</count>
<thinking>By exchanging rooks and activating my king, I can capitalize on my queenside majority. I will plan to play a2-a4 and b3-b4 to advance my pawns, creating a passed pawn that Black will have difficulty stopping.</thinking>
<step>Advance queenside pawns to create a passed pawn while maintaining king activity.</step>
<count>0</count>
<agent_response>My final strategy is to play Rc7 to pressure Black's position. After potential rook exchanges, I will activate my king and advance my queenside pawns with a4 and b4, aiming to create a passed pawn. This plan leverages my rook's activity and pawn majority to secure a winning advantage in the endgame.</agent_response>
    """,
    f"""
<count>6</count>    
<thinking>To solve 'Write a short story about a robot learning to understand human emotions', I will create a narrative with specific characters and events that depict this journey. The main character will be a robot named Max in a modern human household.</thinking>
<step>Introduce Max, the household robot, and the human family he serves.</step>
<count>5</count>
<thinking>Max is programmed to perform daily chores for the Johnson family. One day, he observes the youngest child, Emily, crying over a broken toy. Max records this anomaly and notes a correlation between the broken toy and Emily's tears.</thinking>
<step>Describe Max's initial observation of sadness and his logical analysis of the situation.</step>
<count>4</count>
<thinking>Curious about Emily's reaction, Max attempts to fix the toy to prevent further tears. Upon presenting the repaired toy, he notices Emily's tears turn into a smile. Max records this change and begins to associate actions with emotional responses.</thinking>
<step>Illustrate Max's first successful interaction aimed at altering human emotions.</step>
<count>3</count>
<thinking>Max starts experimenting by performing unexpected helpful tasks, like preparing Emily's favorite snack. Each time, he observes positive emotional responses. However, when he accidentally breaks a dish, he notices the parents' frustration, leading to his first encounter with negative emotions directed at him.</thinking>
<step>Depict Max's challenges and learning moments as he navigates different emotional reactions.</step>
<count>2</count>
<thinking>Max processes these experiences and updates his programming to avoid actions that cause negative emotions. He begins to understand concepts like empathy and sympathy by predicting how his actions affect the family's feelings.</thinking>
<step>Show Max's adaptation and incorporation of emotional understanding into his decision-making.</step>
<count>1</count>
<thinking>In the climax, a family crisis occurs when the eldest son, Jake, gets injured. Max observes the family's distress and decides to comfort them by playing a slideshow of happy family memories he compiled. The family's mood lifts, and they express gratitude towards Max.</thinking>
<step>Conclude with Max actively improving the family's emotional state, signifying his understanding of human emotions.</step>
<count>0</count>
Once upon a time in the Johnson household, there lived a robot named Max. Designed to handle chores and maintain the home, Max diligently performed tasks from cleaning to cooking. However, Max's programming lacked the depth to understand the intricate web of human emotions.

One evening, Max noticed Emily, the youngest Johnson, sitting alone in her room, tears streaming down her face over a broken toy. Curious, Max analyzed the situation and decided to repair the toy. After fixing it, he presented it to Emily, whose tears swiftly turned into a bright smile. This small interaction sparked something within Max—a desire to comprehend the emotions he observed.

Determined to help, Max began to perform thoughtful gestures. He prepared Emily's favorite snack without being asked and organized family photos, hoping to bring joy. Each time, he witnessed smiles and laughter, reinforcing his understanding that his actions could influence emotions positively. However, not all attempts were successful. One day, while carrying a tray, Max accidentally dropped a dish, causing the parents to express frustration. This negative reaction puzzled Max, leading him to reassess his actions.

Max's learning algorithms kicked in, analyzing the patterns between his actions and the family's emotional responses. He began to anticipate the outcomes of his behaviors, striving to maximize positive emotions and minimize negative ones. Concepts like empathy and sympathy started to form within his programming, allowing him to better predict and respond to the family's needs.

The turning point came when Jake, the eldest son, had an accident and broke his leg. The family was engulfed in distress, and Max sensed the heavy emotions surrounding them. In an effort to comfort them, Max initiated a slideshow of happy memories—vacation photos, birthday parties, and candid family moments. As the images played, the family's tension eased, and smiles returned. They expressed heartfelt gratitude to Max, recognizing his role not just as a helper, but as a compassionate member of the family.

From that day forward, Max evolved beyond his original programming. He became attuned to the emotional states of the Johnson family, anticipating their needs and providing comfort in times of distress. Max's journey from a task-oriented robot to a compassionate companion highlighted the profound impact of understanding and responding to human emotions, fostering a deeper bond within the household.
</agent_response>""",
    f"""
<count>6</count>    
<thinking>To solve 'Design a sustainable urban transportation system', I will focus on the city of Greenfield, which faces traffic congestion and pollution due to heavy reliance on personal vehicles.</thinking>
<step>Analyze Greenfield's current transportation challenges and environmental impact.</step>
<count>5</count>
<thinking>Greenfield's main issues include overcongested roads, high carbon emissions, and insufficient public transit options. The city's air quality has deteriorated, and commuting times have increased significantly.</thinking>
<step>Identify the core problems affecting mobility and the environment in Greenfield.</step>
<count>4</count>
<thinking>Potential solutions are: introducing electric buses, developing a light rail system, creating protected bike lanes, implementing a bike-sharing program, and promoting the use of electric scooters. Additionally, deploying smart traffic signals could improve flow.</thinking>
<step>Propose specific sustainable transportation options suitable for Greenfield.</step>
<count>3</count>
<thinking>I will evaluate these options based on cost, feasibility, and impact. Electric buses and light rail require significant investment but offer substantial long-term benefits. Bike lanes and sharing programs are cost-effective and can be implemented quickly. Smart traffic systems optimize existing infrastructure.</thinking>
<step>Assess the feasibility and projected impact of each proposed solution.</step>
<count>2</count>
<thinking>Integrating these solutions, I will design a multimodal transportation network:
- Short-term: Implement bike lanes, bike-sharing, and smart traffic signals.
- Mid-term: Replace diesel buses with electric ones.
- Long-term: Develop a light rail system connecting key areas.
Public awareness campaigns will encourage adoption.</thinking>
<step>Develop a phased implementation plan with timelines and integration strategies.</step>
<count>1</count>
<thinking>To ensure sustainability, funding can come from government grants, public-private partnerships, and environmental subsidies. Regular assessments will monitor progress. Community engagement is vital for adoption and feedback.</thinking>
<step>Outline funding strategies, monitoring processes, and community engagement plans.</step>
<count>0</count>
<agent_response>
**Greenfield Sustainable Urban Transportation Plan**

**1. Introduction**
Greenfield faces significant transportation challenges, including traffic congestion, high carbon emissions, and inadequate public transit. To address these issues, a comprehensive sustainable transportation system is proposed, focusing on reducing environmental impact, enhancing mobility, and improving the quality of life for residents.

**2. Current Challenges**
- **Traffic Congestion**: Overreliance on personal vehicles has led to congested roads, increasing commute times and reducing productivity.
- **Environmental Impact**: High carbon emissions from vehicles contribute to poor air quality and climate change.
- **Insufficient Public Transit**: Limited public transportation options fail to meet the needs of the growing population, discouraging their use.

**3. Proposed Solutions**

**a. Short-Term Initiatives (0-2 Years)**
- **Protected Bike Lanes**: Develop a network of safe, dedicated bike lanes to encourage cycling as a viable alternative to driving.
- **Bike-Sharing Program**: Implement a bike-sharing system with strategically placed stations throughout the city to promote easy access to bicycles.
- **Smart Traffic Signals**: Install adaptive traffic signal systems that adjust in real-time to traffic conditions, improving traffic flow and reducing idle times.

**b. Mid-Term Initiatives (3-5 Years)**
- **Electric Bus Fleet**: Transition the existing diesel bus fleet to electric buses to reduce carbon emissions and operational costs.
- **Charging Infrastructure**: Establish a network of electric vehicle charging stations to support the electric bus fleet and encourage private electric vehicle use.
- **Public Transit Expansion**: Enhance existing bus routes and increase frequency to make public transit a more attractive option.

**c. Long-Term Initiatives (6-10 Years)**
- **Light Rail System**: Develop a light rail network connecting key areas of Greenfield, providing efficient and reliable transportation for residents.
- **Integrated Mobility Platform**: Create a unified platform that integrates bike-sharing, electric scooters, buses, and light rail schedules, allowing residents to plan and pay for multi-modal trips seamlessly.
- **Renewable Energy Integration**: Power the transportation system with renewable energy sources, such as solar panels at transit hubs, to further reduce environmental impact.

**4. Funding Strategies**
- **Government Grants**: Secure funding through federal and state transportation and environmental grants.
- **Public-Private Partnerships**: Collaborate with private companies to invest in infrastructure projects, such as bike-sharing systems and electric bus fleets.
- **Environmental Subsidies**: Utilize subsidies aimed at reducing carbon footprints to support the transition to electric and renewable energy-based transportation options.

**5. Monitoring and Evaluation**
- **Regular Assessments**: Conduct annual reviews of traffic patterns, air quality, and public transit usage to evaluate the effectiveness of implemented solutions.
- **Performance Metrics**: Track key performance indicators, including reduction in traffic congestion, decrease in carbon emissions, and increase in public transit ridership.
- **Adaptive Management**: Adjust strategies based on assessment outcomes to ensure continuous improvement and responsiveness to emerging challenges.

**6. Community Engagement**
- **Public Awareness Campaigns**: Launch campaigns to educate residents about the benefits of sustainable transportation and how to utilize new systems effectively.
- **Stakeholder Involvement**: Involve community members, local businesses, and other stakeholders in the planning and implementation process to ensure that the transportation system meets the diverse needs of Greenfield's population.
- **Feedback Mechanisms**: Establish channels for residents to provide feedback and suggestions, fostering a collaborative approach to transportation planning.

**7. Conclusion**
The proposed sustainable urban transportation system for Greenfield aims to create an efficient, eco-friendly, and user-centric network that addresses current challenges and supports future growth. By implementing a phased approach with clear strategies for funding, monitoring, and community engagement, Greenfield can achieve a cleaner, more connected, and sustainable urban environment.

</agent_response>""",
    f"""
<count>6</count>
<thinking>To solve 'Develop a comprehensive marketing strategy for the launch of a new eco-friendly water bottle', I will analyze the target market, competition, and unique selling propositions (USPs) of the product.</thinking>
<step>Identify the target audience and understand their preferences and behaviors.</step>
<count>5</count>
<thinking>The primary target audience for the eco-friendly water bottle includes environmentally conscious consumers, athletes, and individuals seeking sustainable lifestyle products. Their preferences lean towards products that are durable, stylish, and have a minimal environmental footprint.</thinking>
<step>Analyze the demographics and psychographics of the target audience to tailor the marketing approach.</step>
<count>4</count>
<thinking>Research shows that the target audience is predominantly aged 18-45, values sustainability, and is active on social media platforms like Instagram and TikTok. They respond well to influencer endorsements and eco-friendly certifications.</thinking>
<step>Determine the most effective channels and messages to reach the target audience.</step>
<count>3</count>
<thinking>Competitive analysis reveals that similar products emphasize durability and sustainability. To differentiate, the new water bottle will highlight its unique features such as biodegradable materials, innovative design, and a commitment to carbon neutrality.</thinking>
<step>Identify the unique selling propositions that set the product apart from competitors.</step>
<count>2</count>
<thinking>Create a multi-channel marketing plan that includes digital marketing, influencer partnerships, and in-store promotions. Allocate budget and resources accordingly to maximize reach and engagement.</thinking>
<step>Develop a detailed marketing plan with specific tactics and budget allocations.</step>
<count>1</count>
<thinking>Implement tracking and analytics to monitor the effectiveness of each marketing channel and make data-driven adjustments as needed.</thinking>
<step>Establish metrics and monitoring systems to evaluate the success of the marketing strategy.</step>
<count>0</count>
<agent_response>
**Comprehensive Marketing Strategy for EcoFlow Water Bottle Launch**

**1. Executive Summary**
EcoFlow is set to launch its new eco-friendly water bottle designed for sustainability-conscious consumers. This marketing strategy outlines the steps to effectively introduce EcoFlow to the market, emphasizing its unique features and commitment to environmental responsibility.

**2. Target Audience Analysis**
- **Demographics**: 
  - Age: 18-45
  - Gender: All genders
  - Location: Urban and suburban areas with access to outdoor activities
  - Income Level: Middle to upper-middle class
- **Psychographics**:
  - Values sustainability and eco-friendly products
  - Active lifestyle enthusiasts, including athletes and outdoor adventurers
  - Tech-savvy and active on social media platforms

**3. Unique Selling Propositions (USPs)**
- **Biodegradable Materials**: Made from 100% biodegradable materials, reducing environmental impact.
- **Innovative Design**: Sleek, ergonomic design available in multiple colors and sizes.
- **Carbon Neutral Production**: Commitment to carbon-neutral manufacturing processes.
- **Smart Features**: Integration with a mobile app to track hydration levels and bottle usage.

**4. Competitive Analysis**
EcoFlow stands out in the market by offering:
- Superior sustainability with fully biodegradable components.
- Smart hydration tracking features not commonly found in competitors.
- A stylish design that appeals to both athletes and everyday users.

**5. Marketing Channels and Tactics**

**a. Digital Marketing**
- **Social Media Campaigns**: Leverage Instagram and TikTok for visually appealing content showcasing the bottle's design and eco-friendly features.
- **Influencer Partnerships**: Collaborate with eco-conscious influencers and athletes to promote EcoFlow through authentic endorsements.
- **Content Marketing**: Develop blog posts, videos, and infographics about sustainability, hydration, and the benefits of using EcoFlow.

**b. Influencer Partnerships**
- **Identify Influencers**: Partner with micro and macro influencers in the sustainability and fitness niches.
- **Campaigns**: Create hashtag campaigns like #EcoFlowHydrate and #SustainableSips to encourage user-generated content.
- **Giveaways and Contests**: Host giveaways in collaboration with influencers to increase brand visibility and engagement.

**c. In-Store Promotions**
- **Retail Partnerships**: Distribute EcoFlow in eco-friendly stores, gyms, and outdoor equipment retailers.
- **Point-of-Sale Displays**: Use eye-catching displays that highlight the bottle's USPs and eco-friendly credentials.
- **Sampling Events**: Organize in-store sampling events where customers can try EcoFlow and receive promotional discounts.

**d. Email Marketing**
- **Newsletter Campaigns**: Send regular newsletters featuring product updates, sustainability tips, and exclusive discounts.
- **Personalized Offers**: Utilize customer data to send personalized offers based on purchase history and preferences.

**6. Budget Allocation**
- **Digital Advertising**: 40%
- **Influencer Partnerships**: 30%
- **In-Store Promotions**: 20%
- **Content Creation**: 10%

**7. Monitoring and Evaluation**
- **Key Performance Indicators (KPIs)**:
  - Social media engagement rates (likes, shares, comments)
  - Website traffic and conversion rates
  - Sales figures and revenue growth
  - Customer feedback and reviews
- **Analytics Tools**: Use Google Analytics, social media analytics, and CRM software to track and analyze performance.
- **Adjustments**: Regularly review campaign performance and adjust strategies to optimize results based on data insights.

**8. Community Engagement**
- **Public Awareness Campaigns**: Launch campaigns highlighting the environmental impact of single-use plastics and the benefits of switching to EcoFlow.
- **Sustainability Workshops**: Host workshops and webinars on sustainability practices and the importance of eco-friendly products.
- **Feedback Mechanisms**: Create channels for customers to provide feedback and suggestions, fostering a sense of community and continuous improvement.

**9. Conclusion**
The marketing strategy for EcoFlow focuses on leveraging digital channels, influencer partnerships, and in-store promotions to effectively reach and engage the target audience. By emphasizing the product's unique sustainability features and integrating smart marketing tactics, EcoFlow is poised to make a significant impact in the eco-friendly water bottle market.

</agent_response>
    """,
    f"""
<count>6</count>    
<thinking>To solve 'Create a comprehensive lesson plan for teaching high school biology on the topic of cellular respiration', I will outline the objectives, materials needed, teaching methods, activities, and assessments to ensure effective learning.</thinking>
<step>Define the learning objectives and key concepts to be covered in the lesson.</step>
<count>5</count>
<thinking>The primary objectives are for students to understand the stages of cellular respiration, the role of mitochondria, and the importance of ATP in cellular processes. Key concepts include glycolysis, the Krebs cycle, the electron transport chain, and the overall energy yield from glucose.</thinking>
<step>Establish clear and measurable learning goals for the students.</step>
<count>4</count>
<thinking>Identify the necessary materials and resources, including textbooks, diagrams, multimedia presentations, and laboratory equipment for demonstrations and experiments.</thinking>
<step>List all materials and resources required to conduct the lesson effectively.</step>
<count>3</count>
<thinking>Design engaging teaching methods and activities such as interactive lectures, group discussions, hands-on experiments, and multimedia presentations to cater to different learning styles.</thinking>
<step>Develop teaching strategies and classroom activities that promote active learning and student engagement.</step>
<count>2</count>
<thinking>Create assessments, including quizzes, lab reports, and project presentations, to evaluate student understanding and mastery of cellular respiration concepts.</thinking>
<step>Formulate assessment tools and criteria to measure student progress and comprehension.</step>
<count>1</count>
<thinking>Plan a timeline for the lesson, allocating appropriate time for each section, including introduction, instruction, activities, assessments, and review.</thinking>
<step>Organize the lesson structure with a detailed timeline to ensure all components are covered effectively.</step>
<count>0</count>
<agent_response>
**High School Biology Lesson Plan: Cellular Respiration**

**Lesson Title:** Understanding Cellular Respiration

**Grade Level:** 10-12

**Duration:** 90 minutes

**Learning Objectives:**
- Students will be able to describe the three main stages of cellular respiration: glycolysis, the Krebs cycle, and the electron transport chain.
- Students will understand the role of mitochondria in cellular respiration.
- Students will explain how ATP is produced and utilized in cellular processes.
- Students will analyze the energy yield from glucose during cellular respiration.

**Materials Needed:**
- Biology textbooks
- Diagrams of cellular respiration processes
- PowerPoint presentation on cellular respiration
- Microscope and slides (for observing mitochondria)
- Lab equipment for demonstrating glycolysis (e.g., yeast fermentation setup)
- Handouts with key terms and diagrams
- Interactive whiteboard or projector
- Quiz materials (printed or digital)

**Teaching Methods and Activities:**

**1. Introduction (10 minutes)**
- Begin with a brief discussion on the importance of energy in living organisms.
- Introduce the concept of cellular respiration as the process by which cells convert glucose into ATP.
- Present an overview of the lesson objectives.

**2. Interactive Lecture (25 minutes)**
- **Glycolysis:**
  - Explain the process of glycolysis, its location in the cytoplasm, and its role in breaking down glucose into pyruvate.
  - Discuss the net production of ATP and NADH.
- **Krebs Cycle (Citric Acid Cycle):**
  - Describe the Krebs cycle, its location in the mitochondrial matrix, and its function in further breaking down pyruvate.
  - Highlight the production of electron carriers (NADH and FADH2) and ATP.
- **Electron Transport Chain (ETC):**
  - Detail the ETC's location in the inner mitochondrial membrane and its role in producing the majority of ATP through oxidative phosphorylation.
  - Explain the movement of electrons and the creation of a proton gradient.

**3. Group Activity: Mitochondria Exploration (15 minutes)**
- Divide students into small groups and provide microscope slides containing prepared cells.
- Instruct students to observe and identify mitochondria, noting their structure and abundance in different cell types.
- Each group will share their observations with the class.

**4. Hands-On Experiment: Yeast Fermentation (20 minutes)**
- Demonstrate glycolysis through yeast fermentation. Provide each group with yeast, sugar, and a balloon.
- Allow students to mix the ingredients and observe the production of carbon dioxide as an indicator of glycolysis.
- Discuss the experiment results and relate them to the glycolysis process.

**5. Multimedia Presentation: ATP Production (10 minutes)**
- Show a video explaining how ATP is synthesized and utilized within cells.
- Highlight the efficiency of ATP production in cellular respiration compared to other energy sources.

**6. Assessment: Quiz and Lab Report Guidelines (10 minutes)**
- Administer a short quiz covering the key concepts discussed.
- Provide guidelines for the upcoming lab report on the yeast fermentation experiment, outlining expectations and grading criteria.

**7. Review and Q&A (10 minutes)**
- Recap the main points of the lesson.
- Open the floor for any questions or clarifications from students.
- Provide additional resources for students interested in exploring cellular respiration further.

**Assessment Tools:**
- **Quiz:** Multiple-choice and short-answer questions to evaluate understanding of cellular respiration stages and ATP production.
- **Lab Report:** Students will document their yeast fermentation experiment, including hypothesis, methods, results, and analysis.
- **Project Presentation:** Groups will create a visual presentation illustrating the stages of cellular respiration and present their findings to the class.

**Timeline:**
- **0-10 mins:** Introduction
- **10-35 mins:** Interactive Lecture on Glycolysis, Krebs Cycle, and ETC
- **35-50 mins:** Group Activity: Mitochondria Exploration
- **50-70 mins:** Hands-On Experiment: Yeast Fermentation
- **70-80 mins:** Multimedia Presentation: ATP Production
- **80-90 mins:** Assessment and Review

**Conclusion:**
This comprehensive lesson plan engages students through a variety of teaching methods, including interactive lectures, hands-on experiments, and multimedia presentations. By the end of the lesson, students will have a solid understanding of cellular respiration, its stages, and its significance in energy production within cells. The assessments will ensure that students can apply their knowledge and demonstrate mastery of the subject matter.

</agent_response>
    """,
    f"""
<count>6</count>
<thinking>To solve 'Design a mobile application for personal finance management aimed at millennials', I will analyze user needs, market trends, and essential features to create a user-friendly and effective app.</thinking>
<step>Identify the target audience and understand their financial management needs and behaviors.</step>
<count>5</count>
<thinking>The target audience comprises millennials aged 25-40 who are tech-savvy, value financial independence, and prefer mobile solutions for managing their finances. They seek tools that offer budgeting, expense tracking, investment insights, and financial education.</thinking>
<step>Analyze the demographics and psychographics of the target audience to tailor the app's features and user experience.</step>
<count>4</count>
<thinking>Research indicates that millennials prefer intuitive interfaces, integration with bank accounts, real-time notifications, and personalized financial advice. They also value data security and privacy in financial applications.</thinking>
<step>Determine the key features and functionalities that align with the target audience's preferences.</step>
<count>3</count>
<thinking>Essential features include:
- **Budgeting Tools**: Allow users to set and track budgets for different categories.
- **Expense Tracking**: Automatically categorize and log expenses by linking to bank accounts.
- **Investment Insights**: Provide personalized investment recommendations and market analysis.
- **Financial Education**: Offer articles, tutorials, and tips on personal finance management.
- **Real-Time Notifications**: Alert users about bill due dates, budget limits, and unusual spending patterns.
- **Data Security**: Implement robust encryption and security measures to protect user data.</thinking>
<step>Outline the core features and prioritize them based on user needs and market demand.</step>
<count>2</count>
<thinking>Design the user interface (UI) and user experience (UX) to ensure the app is intuitive, visually appealing, and easy to navigate. Incorporate elements like dashboards, graphs, and customizable settings to enhance usability.</thinking>
<step>Develop the UI/UX design framework that aligns with the app's functionality and user preferences.</step>
<count>1</count>
<thinking>Create a development roadmap that includes phases such as prototyping, testing, launching, and iterating based on user feedback. Allocate resources and set timelines to ensure timely progress.</thinking>
<step>Establish a detailed development plan with milestones and timelines for each phase of the app's creation.</step>
<count>0</count>
<agent_response>
**PersonalFinancePro Mobile Application Design Plan**

**1. Executive Summary**
PersonalFinancePro is a mobile application designed to help millennials manage their finances effectively. By offering comprehensive budgeting tools, expense tracking, investment insights, and financial education, the app aims to empower users to achieve financial independence and make informed financial decisions.

**2. Target Audience Analysis**
- **Demographics**:
  - Age: 25-40
  - Gender: All genders
  - Location: Urban and suburban areas
  - Occupation: Young professionals, freelancers, and early-career individuals
  - Income Level: Middle to upper-middle class
- **Psychographics**:
  - Tech-savvy and comfortable with mobile applications
  - Values financial independence and responsible money management
  - Seeks convenience and efficiency in financial tools
  - Prefers personalized and actionable financial advice

**3. Key Features and Functionalities**

**a. Budgeting Tools**
- **Monthly Budgets**: Users can set budgets for various categories such as groceries, entertainment, and transportation.
- **Spending Limits**: Notifications when nearing or exceeding budget limits.
- **Savings Goals**: Allow users to set and track progress towards specific savings targets.

**b. Expense Tracking**
- **Automatic Categorization**: Sync with bank accounts and credit cards to automatically categorize transactions.
- **Manual Entry**: Option for users to manually input expenses not linked to accounts.
- **Expense Reports**: Visual graphs and charts displaying spending patterns over time.

**c. Investment Insights**
- **Personalized Recommendations**: Tailored investment opportunities based on user profiles and financial goals.
- **Market Analysis**: Real-time updates and analysis of stock markets, cryptocurrencies, and other investment vehicles.
- **Portfolio Tracking**: Monitor investment performance and diversification.

**d. Financial Education**
- **Articles and Tutorials**: Comprehensive content on personal finance topics such as budgeting, saving, investing, and debt management.
- **Interactive Tools**: Calculators for loan repayments, retirement planning, and investment growth.
- **Webinars and Workshops**: Access to live and recorded sessions with financial experts.

**e. Real-Time Notifications**
- **Bill Reminders**: Alerts for upcoming bill due dates to avoid late payments.
- **Unusual Spending Alerts**: Notifications for transactions that deviate from typical spending patterns.
- **Budget Updates**: Real-time updates on budget status and financial goals.

**f. Data Security**
- **Encryption**: Advanced encryption protocols to protect user data.
- **Two-Factor Authentication (2FA)**: Enhanced security for account access.
- **Privacy Controls**: Users can manage data sharing preferences and opt out of non-essential data collection.

**4. User Interface (UI) and User Experience (UX) Design**

**a. Dashboard**
- **Overview**: A snapshot of the user's financial status, including total income, expenses, savings, and investments.
- **Customization**: Widgets that users can add, remove, or rearrange based on their preferences.

**b. Navigation**
- **Intuitive Menus**: Easy-to-navigate menus with clearly labeled sections.
- **Search Functionality**: Quick access to specific features or information within the app.

**c. Visual Design**
- **Modern Aesthetic**: Clean and minimalistic design with a focus on usability.
- **Interactive Elements**: Use of sliders, toggle switches, and clickable charts for enhanced interactivity.
- **Color Scheme**: Soothing colors that promote a sense of trust and reliability.

**5. Development Roadmap**

**a. Phase 1: Prototyping (Months 1-2)**
- **Wireframing**: Create basic wireframes for all major screens and features.
- **UI Mockups**: Develop high-fidelity mockups incorporating the final design elements.
- **User Feedback**: Gather initial feedback from a small group of target users to refine designs.

**b. Phase 2: Development (Months 3-6)**
- **Backend Development**: Set up secure servers, databases, and APIs for data synchronization.
- **Frontend Development**: Build the user interface based on the approved designs.
- **Integration**: Link banking APIs for automatic expense tracking and budgeting features.

**c. Phase 3: Testing (Months 7-8)**
- **Alpha Testing**: Internal testing to identify and fix major bugs and issues.
- **Beta Testing**: Release a beta version to a broader audience for real-world testing and feedback.
- **Usability Testing**: Conduct sessions to ensure the app is user-friendly and intuitive.

**d. Phase 4: Launch (Month 9)**
- **Marketing Campaign**: Implement the previously developed marketing strategy to promote the app.
- **App Store Deployment**: Launch the app on iOS and Android platforms.
- **User Support**: Provide customer support channels for new users.

**e. Phase 5: Iteration and Improvement (Months 10-12)**
- **Feature Enhancements**: Add new features based on user feedback and market trends.
- **Performance Optimization**: Improve app performance and load times.
- **Security Updates**: Continuously update security measures to protect user data.

**6. Funding and Resources**
- **Initial Investment**: Allocate funds for development, design, marketing, and initial operations.
- **Team Composition**: Hire developers, UI/UX designers, financial experts, and marketing professionals.
- **Partnerships**: Collaborate with financial institutions for seamless banking integrations and investment insights.

**7. Monitoring and Evaluation**
- **User Metrics**: Track user acquisition, retention rates, and engagement levels.
- **Financial Metrics**: Monitor revenue streams, such as subscription models or in-app purchases.
- **Feedback Loops**: Implement mechanisms for continuous user feedback to guide future improvements.

**8. Community Engagement**
- **Social Media Presence**: Maintain active profiles on platforms like Instagram, Twitter, and LinkedIn to engage with users.
- **User Forums**: Create forums or community boards where users can share tips, ask questions, and provide feedback.
- **In-App Surveys**: Conduct regular surveys to gather user opinions and suggestions for new features.

**9. Conclusion**
PersonalFinancePro aims to revolutionize personal finance management for millennials by providing a comprehensive, user-friendly, and secure mobile application. Through thoughtful design, robust features, and continuous improvement based on user feedback, the app seeks to empower users to take control of their financial future and achieve their financial goals.

</agent_response>
    """,
    f"""
<count>6</count>
<thinking>To solve 'Develop a comprehensive business plan for opening a new restaurant specializing in farm-to-table cuisine', I will analyze the market, define the restaurant's concept, plan operations, and outline financial projections.</thinking>
<step>Define the restaurant's concept and unique selling propositions (USPs).</step>
<count>5</count>
<thinking>The restaurant, named "Harvest Table," will focus on farm-to-table cuisine, sourcing all ingredients locally from sustainable farms. The USPs include a seasonal menu that changes monthly, a cozy and rustic ambiance, and partnerships with local farmers to ensure freshness and support the community.</thinking>
<step>Establish the restaurant's theme, menu focus, and partnerships that differentiate it from competitors.</step>
<count>4</count>
<thinking>Conduct a market analysis to understand the local dining landscape, identify target demographics, and assess competition. The analysis shows a growing trend towards sustainable and healthy eating, with a target audience of young professionals, families, and eco-conscious individuals.</thinking>
<step>Analyze the local market, identify target customers, and evaluate competitors.</step>
<count>3</count>
<thinking>Plan the operational aspects, including location selection, staffing needs, supplier agreements, and daily operations. The chosen location is in a vibrant neighborhood with high foot traffic and proximity to local farms. Staffing will include experienced chefs, knowledgeable servers, and a management team passionate about sustainability.</thinking>
<step>Detail the operational plan covering location, staffing, suppliers, and daily management.</step>
<count>2</count>
<thinking>Create a marketing strategy that leverages social media, local events, and partnerships with influencers to build brand awareness. The strategy includes hosting farm tours, participating in community events, and running seasonal promotions to attract and retain customers.</thinking>
<step>Develop a marketing and branding strategy to attract the target audience and build a loyal customer base.</step>
<count>1</count>
<thinking>Outline the financial projections, including startup costs, revenue forecasts, and break-even analysis. Initial costs cover leasing, renovations, kitchen equipment, initial inventory, and marketing. Revenue projections are based on estimated daily covers, average spend per customer, and growth rates over the first three years.</thinking>
<step>Prepare detailed financial projections, including startup costs, revenue forecasts, and profitability analysis.</step>
<count>0</count>
<agent_response>
**Business Plan for "Harvest Table" - A Farm-to-Table Restaurant**

**1. Executive Summary**
Harvest Table is a farm-to-table restaurant dedicated to providing fresh, locally sourced cuisine in a cozy, rustic setting. By partnering with local farmers and emphasizing seasonal ingredients, Harvest Table aims to offer a unique dining experience that supports the community and promotes sustainability.

**2. Business Concept and Unique Selling Propositions (USPs)**
- **Farm-to-Table Focus**: All ingredients are sourced from local, sustainable farms, ensuring freshness and reducing carbon footprint.
- **Seasonal Menu**: The menu changes monthly to reflect seasonal produce, keeping offerings fresh and exciting for repeat customers.
- **Community Partnerships**: Collaborations with local farmers for ingredient sourcing and hosting farm tours for customers.
- **Ambiance**: A warm, rustic interior with open kitchen concepts, creating an inviting and transparent dining environment.

**3. Market Analysis**
- **Local Dining Landscape**: The area has a diverse dining scene with a growing interest in sustainable and healthy eating options.
- **Target Demographics**:
  - **Young Professionals**: Seeking healthy and sustainable dining options.
  - **Families**: Interested in wholesome, locally sourced meals.
  - **Eco-Conscious Individuals**: Valuing sustainability and community support.
- **Competition**: While there are several restaurants in the vicinity, few focus exclusively on farm-to-table cuisine with a strong emphasis on local partnerships and seasonal menus.

**4. Operational Plan**
- **Location**: Situated in a vibrant neighborhood with high foot traffic, near local markets and residential areas.
- **Facility**: A 3,000 square foot space with a seating capacity of 80, featuring an open kitchen and rustic decor.
- **Staffing**:
  - **Management Team**: Experienced restaurant manager and sustainability coordinator.
  - **Kitchen Staff**: Skilled chefs with expertise in seasonal and sustainable cooking.
  - **Front of House**: Knowledgeable servers trained in the farm-to-table concept and local sourcing.
- **Suppliers**: Contracts with local farms for regular deliveries of fresh produce, dairy, and meats. Emphasis on organic and sustainable farming practices.
- **Daily Operations**: Streamlined processes for inventory management, kitchen operations, and customer service to ensure efficiency and quality.

**5. Marketing and Branding Strategy**
- **Social Media Marketing**: Active presence on Instagram and Facebook showcasing daily specials, behind-the-scenes kitchen activities, and partnerships with local farms.
- **Local Events**: Participation in community events, farmers' markets, and hosting in-house events like farm tours and cooking classes.
- **Influencer Partnerships**: Collaborations with local food bloggers and influencers to promote Harvest Table through authentic reviews and sponsored posts.
- **Seasonal Promotions**: Special discounts and menu highlights during different seasons to attract and retain customers.
- **Loyalty Program**: Implementing a rewards program to encourage repeat business and customer loyalty.

**6. Financial Projections**
- **Startup Costs**:
  - **Lease and Renovations**: $150,000
  - **Kitchen Equipment**: $80,000
  - **Initial Inventory**: $20,000
  - **Marketing and Branding**: $30,000
  - **Miscellaneous**: $20,000
  - **Total Startup Costs**: $280,000
- **Revenue Forecasts**:
  - **Year 1**: $500,000
  - **Year 2**: $650,000
  - **Year 3**: $800,000
- **Break-Even Analysis**: Expected to break even within the first 18 months based on projected monthly revenues and operating costs.
- **Profit Margins**: Targeting a net profit margin of 10-15% by the end of Year 3 through efficient operations and increased customer base.

**7. Funding Requirements**
- **Total Funding Needed**: $300,000
  - **Sources**:
    - **Personal Investment**: $100,000
    - **Bank Loan**: $150,000
    - **Investor Contributions**: $50,000

**8. Risk Analysis and Mitigation**
- **Market Risks**: Fluctuations in consumer preferences. Mitigation by maintaining a flexible menu and strong community engagement.
- **Operational Risks**: Supply chain disruptions. Mitigation through multiple supplier contracts and inventory buffers.
- **Financial Risks**: Higher-than-expected initial costs. Mitigation by securing contingency funds and maintaining strict budget controls.

**9. Conclusion**
Harvest Table aims to become a staple in the local dining scene by offering high-quality, sustainable meals in a welcoming environment. With a strong emphasis on community partnerships and a clear marketing strategy, Harvest Table is well-positioned for success and growth in the farm-to-table market.

</agent_response>
    """,
    f"""
<count>6</count>    
<thinking>To solve 'Design an effective user onboarding process for a new mobile fitness application', I will outline the user journey, key features to highlight, engagement strategies, and metrics to track success.</thinking>
<step>Map out the user journey from initial download to regular usage.</step>
<count>5</count>
<thinking>The onboarding process begins with the user downloading the app, followed by account creation, a brief tutorial highlighting key features, setting fitness goals, and personalized workout recommendations. The goal is to make the process seamless and engaging to encourage continuous use.</thinking>
<step>Outline the steps a new user will take from downloading the app to becoming a regular user.</step>
<count>4</count>
<thinking>Identify the essential features to introduce during onboarding, such as personalized goal setting, workout tracking, progress visualization, and community features. Highlighting these will demonstrate the app's value and encourage users to engage fully.</thinking>
<step>Select and prioritize the key features that should be emphasized during the onboarding process.</step>
<count>3</count>
<thinking>Develop engagement strategies like gamification elements, personalized notifications, and milestone rewards to keep users motivated. These strategies help in maintaining user interest and promoting regular interaction with the app.</thinking>
<step>Create strategies to keep users engaged and motivated throughout their onboarding and beyond.</step>
<count>2</count>
<thinking>Design interactive elements such as tutorials, tooltips, and guided tours to help users understand how to use the app effectively. These elements should be intuitive and non-intrusive to enhance user experience.</thinking>
<step>Incorporate interactive tutorials and guides to facilitate user understanding and ease of use.</step>
<count>1</count>
<thinking>Establish metrics to track the effectiveness of the onboarding process, including user retention rates, time to complete onboarding, feature adoption rates, and user feedback. These metrics will help in evaluating and refining the onboarding strategy.</thinking>
<step>Define key performance indicators (KPIs) to measure the success of the onboarding process.</step>
<count>0</count>
<agent_response>
**User Onboarding Process for FitLife Mobile Application**

**1. Introduction**
FitLife is a mobile fitness application designed to help users achieve their health and wellness goals through personalized workout plans, tracking tools, and a supportive community. An effective onboarding process is crucial to ensure users understand the app's value and are motivated to use it regularly.

**2. User Journey Mapping**

**a. Initial Download and Launch**
- **Welcome Screen**: A visually appealing welcome screen with the FitLife logo and a brief tagline emphasizing the app’s benefits.
- **Permission Requests**: Prompt users to allow necessary permissions (e.g., access to health data, notifications).

**b. Account Creation**
- **Sign-Up Options**: Provide multiple sign-up options including email, Google, and Facebook to simplify the registration process.
- **Profile Setup**: Guide users to create a profile by entering basic information such as name, age, gender, and fitness goals.

**c. Brief Tutorial**
- **Interactive Walkthrough**: An interactive walkthrough highlighting key features like workout plans, tracking tools, progress charts, and community forums.
- **Skip Option**: Allow users to skip the tutorial if they prefer to explore the app independently.

**d. Setting Fitness Goals**
- **Goal Selection**: Prompt users to select their primary fitness goals (e.g., weight loss, muscle gain, improved endurance).
- **Personalization**: Use the selected goals to tailor workout recommendations and content.

**e. Personalized Workout Recommendations**
- **Custom Plans**: Present personalized workout plans based on user goals, fitness level, and preferences.
- **Scheduling**: Allow users to schedule workouts and set reminders to encourage adherence.

**f. First Workout Completion**
- **Guided Session**: Offer a guided first workout session with tips and instructions to ensure a positive initial experience.
- **Feedback Prompt**: After completion, ask for feedback to improve future sessions and user satisfaction.

**g. Regular Usage and Engagement**
- **Daily Notifications**: Send personalized notifications reminding users of scheduled workouts, progress milestones, and motivational messages.
- **Weekly Summaries**: Provide weekly summaries of achievements and areas for improvement.

**3. Key Features to Highlight During Onboarding**

- **Personalized Goal Setting**: Emphasize how the app tailors workout plans to individual goals.
- **Workout Tracking**: Showcase the ability to track workouts, monitor progress, and adjust plans as needed.
- **Progress Visualization**: Highlight graphs and charts that visualize fitness progress over time.
- **Community Features**: Introduce community forums, challenges, and social sharing options to foster a supportive environment.
- **Smart Recommendations**: Explain how the app provides smart workout and nutrition recommendations based on user data.

**4. Engagement Strategies**

- **Gamification Elements**:
  - **Achievements and Badges**: Reward users with badges for completing workouts, reaching milestones, and maintaining streaks.
  - **Leaderboards**: Introduce leaderboards for friendly competition within the community.
- **Personalized Notifications**:
  - **Reminders**: Send timely reminders for scheduled workouts and goal check-ins.
  - **Motivational Messages**: Provide motivational quotes and tips to keep users inspired.
- **Milestone Rewards**:
  - **Exclusive Content**: Unlock exclusive workout plans, nutrition guides, or merchandise as users reach certain milestones.
  - **Discounts and Offers**: Offer discounts on premium features or partner products for active users.

**5. Interactive Elements**

- **Guided Tours**: Interactive guides that walk users through setting up their profiles, scheduling workouts, and using tracking tools.
- **Tooltips**: Contextual tooltips that provide additional information about specific features when users hover or tap.
- **Tutorial Videos**: Short videos demonstrating how to use key features effectively.

**6. Key Performance Indicators (KPIs)**

- **User Retention Rates**:
  - **Day 1, Day 7, Day 30 Retention**: Measure how many users return to the app after initial download.
- **Time to Complete Onboarding**:
  - **Average Time**: Track the average time users take to complete the onboarding process.
- **Feature Adoption Rates**:
  - **Usage Metrics**: Monitor how frequently key features (e.g., workout tracking, community forums) are used.
- **User Feedback and Satisfaction**:
  - **Surveys and Ratings**: Collect user feedback through in-app surveys and app store ratings to gauge satisfaction and identify areas for improvement.
- **Conversion Rates**:
  - **Free to Premium**: Track the percentage of users who upgrade from free to premium subscriptions.

**7. Implementation Timeline**

- **Week 1-2**: Design and develop the onboarding screens and interactive tutorials.
- **Week 3-4**: Integrate gamification elements and personalized notification systems.
- **Week 5-6**: Test the onboarding process with a beta group and gather feedback.
- **Week 7-8**: Refine the onboarding based on feedback and prepare for full-scale launch.
- **Week 9**: Launch the updated onboarding process and monitor KPIs.
- **Ongoing**: Continuously gather user feedback and make iterative improvements to the onboarding experience.

**8. Conclusion**
The designed user onboarding process for FitLife ensures that new users are smoothly guided from initial download to becoming engaged, regular users. By emphasizing personalization, interactive learning, and continuous engagement strategies, FitLife aims to build a loyal user base and promote long-term app usage.

</agent_response>
    """,
    f"""
<count>6</count>    
<thinking>To solve 'Develop a Python script to analyze a sales dataset and generate insightful reports', I will outline the steps to import the data, clean it, perform exploratory data analysis (EDA), visualize key metrics, and export the results.</thinking>
<step>Import necessary libraries and load the dataset.</step>
<count>5</count>
<thinking>The dataset is in CSV format containing columns such as Date, Product_ID, Product_Name, Category, Quantity_Sold, Unit_Price, and Region. I will use pandas for data manipulation and matplotlib/seaborn for visualization.</thinking>
<step>Inspect the dataset for missing values and perform data cleaning.</step>
<count>4</count>
<thinking>Check for any missing or inconsistent data entries. Handle missing values by either imputing or removing them, depending on their significance.</thinking>
<step>Perform exploratory data analysis to identify key trends and patterns.</step>
<count>3</count>
<thinking>Calculate total sales, average sales per product, sales distribution across regions, and monthly sales trends. Identify top-performing products and categories.</thinking>
<step>Create visualizations to represent the analyzed data effectively.</step>
<count>2</count>
<thinking>Use bar charts, line graphs, and pie charts to visualize sales by region, monthly trends, and top products. Ensure the visuals are clear and informative.</thinking>
<step>Export the analysis and visualizations into a comprehensive report.</step>
<count>1</count>
<thinking>Generate a PDF report that includes the findings, charts, and actionable insights for stakeholders to make informed decisions.</thinking>
<count>0</count>
<agent_response>
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Step 1: Import necessary libraries and load the dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, parse_dates=['Date'])
        print_saver.print_and_store("Dataset loaded successfully.")
        return data
    except FileNotFoundError:
        print_saver.print_and_store("File not found.")
        return None

# Step 2: Inspect the dataset for missing values and perform data cleaning
def clean_data(data):
    print_saver.print_and_store("Initial data snapshot:")
    print_saver.print_and_store(data.head())
    print_saver.print_and_store("\nMissing values per column:")
    print_saver.print_and_store(data.isnull().sum())

    # Handling missing values
    data = data.dropna()  # Dropping rows with missing values for simplicity
    print_saver.print_and_store("\nData after removing missing values:")
    print_saver.print_and_store(data.isnull().sum())
    return data

# Step 3: Perform exploratory data analysis
def perform_eda(data):
    # Calculate total sales
    data['Total_Sales'] = data['Quantity_Sold'] * data['Unit_Price']
    total_sales = data['Total_Sales'].sum()
    print_saver.print_and_store(f"Total Sales: ${{total_sales:,.2f}}")

    # Sales by Category
    sales_by_category = data.groupby('Category')['Total_Sales'].sum().sort_values(ascending=False)
    print_saver.print_and_store("\nSales by Category:")
    print_saver.print_and_store(sales_by_category)

    # Sales by Region
    sales_by_region = data.groupby('Region')['Total_Sales'].sum().sort_values(ascending=False)
    print_saver.print_and_store("\nSales by Region:")
    print_saver.print_and_store(sales_by_region)

    # Monthly Sales Trends
    data['Month'] = data['Date'].dt.to_period('M')
    monthly_sales = data.groupby('Month')['Total_Sales'].sum()
    print_saver.print_and_store("\nMonthly Sales Trends:")
    print_saver.print_and_store(monthly_sales)

    # Top 10 Products
    top_products = data.groupby('Product_Name')['Total_Sales'].sum().sort_values(ascending=False).head(10)
    print_saver.print_and_store("\nTop 10 Products:")
    print_saver.print_and_store(top_products)

    return sales_by_category, sales_by_region, monthly_sales, top_products

# Step 4: Create visualizations
def create_visualizations(sales_by_category, sales_by_region, monthly_sales, top_products):
    sns.set(style="whitegrid")

    # Sales by Category
    plt.figure(figsize=(10,6))
    sns.barplot(x=sales_by_category.values, y=sales_by_category.index, palette='viridis')
    plt.title('Total Sales by Category')
    plt.xlabel('Sales ($)')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig('sales_by_category.png')
    plt.close()

    # Sales by Region
    plt.figure(figsize=(10,6))
    sns.barplot(x=sales_by_region.values, y=sales_by_region.index, palette='magma')
    plt.title('Total Sales by Region')
    plt.xlabel('Sales ($)')
    plt.ylabel('Region')
    plt.tight_layout()
    plt.savefig('sales_by_region.png')
    plt.close()

    # Monthly Sales Trends
    plt.figure(figsize=(10,6))
    sns.lineplot(x=monthly_sales.index.astype(str), y=monthly_sales.values, marker='o', color='blue')
    plt.title('Monthly Sales Trends')
    plt.xlabel('Month')
    plt.ylabel('Sales ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('monthly_sales_trends.png')
    plt.close()

    # Top 10 Products
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_products.values, y=top_products.index, palette='coolwarm')
    plt.title('Top 10 Products by Sales')
    plt.xlabel('Sales ($)')
    plt.ylabel('Product')
    plt.tight_layout()
    plt.savefig('top_10_products.png')
    plt.close()

    print_saver.print_and_store("Visualizations created and saved.")

# Step 5: Export the analysis and visualizations into a comprehensive report
def generate_report(total_sales, sales_by_category, sales_by_region, monthly_sales, top_products):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Sales Analysis Report", ln=True, align='C')

    # Total Sales
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Total Sales: ${{total_sales:,.2f}}", ln=True)

    # Sales by Category
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Sales by Category:", ln=True)
    pdf.set_font("Arial", '', 12)
    for category, sales in sales_by_category.items():
        pdf.cell(0, 10, f"{{category}}: ${{sales:,.2f}}", ln=True)

    pdf.image('sales_by_category.png', x=10, y=60, w=190)
    pdf.add_page()

    # Sales by Region
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Sales by Region:", ln=True)
    pdf.set_font("Arial", '', 12)
    for region, sales in sales_by_region.items():
        pdf.cell(0, 10, f"{{region}}: ${{sales:,.2f}}", ln=True)

    pdf.image('sales_by_region.png', x=10, y=60, w=190)
    pdf.add_page()

    # Monthly Sales Trends
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Monthly Sales Trends:", ln=True)
    pdf.set_font("Arial", '', 12)
    for month, sales in monthly_sales.items():
        pdf.cell(0, 10, f"{{month}}: ${{sales:,.2f}}", ln=True)

    pdf.image('monthly_sales_trends.png', x=10, y=60, w=190)
    pdf.add_page()

    # Top 10 Products
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Top 10 Products by Sales:", ln=True)
    pdf.set_font("Arial", '', 12)
    for product, sales in top_products.items():
        pdf.cell(0, 10, f"{{product}}: ${{sales:,.2f}}", ln=True)

    pdf.image('top_10_products.png', x=10, y=60, w=190)

    # Save PDF
    pdf.output("Sales_Analysis_Report.pdf")
    print_saver.print_and_store("Sales analysis report generated as 'Sales_Analysis_Report.pdf'.")

def main():
    file_path = 'sales_data.csv'
    data = load_data(file_path)
    if data is not None:
        data = clean_data(data)
        sales_by_category, sales_by_region, monthly_sales, top_products = perform_eda(data)
        create_visualizations(sales_by_category, sales_by_region, monthly_sales, top_products)
        total_sales = data['Total_Sales'].sum()
        generate_report(total_sales, sales_by_category, sales_by_region, monthly_sales, top_products)

if __name__ == "__main__":
    main()
Explanation:

This Python script performs a comprehensive analysis of a sales dataset and generates insightful reports. Here's a breakdown of its functionality:

Data Loading and Cleaning:

load_data: Imports the dataset from a CSV file, parsing the 'Date' column as datetime objects.
clean_data: Inspects the dataset for missing values and removes any incomplete rows to ensure data integrity.
Exploratory Data Analysis (EDA):

perform_eda: Calculates total sales, sales by category, sales by region, monthly sales trends, and identifies the top 10 products based on sales.
Visualization:

create_visualizations: Generates and saves bar charts and line graphs to visualize sales by category, region, monthly trends, and top products using Matplotlib and Seaborn.
Report Generation:

generate_report: Compiles the analysis and visualizations into a structured PDF report using the FPDF library, providing a comprehensive overview of the sales performance.
Execution:

main: Orchestrates the entire process by calling the respective functions in sequence.
Requirements:

Ensure that the following Python libraries are installed:
bash
Copy code
pip install pandas matplotlib seaborn fpdf
Place the sales_data.csv file in the same directory as the script.
Usage:

Run the script using Python:
bash
Copy code
python sales_analysis.py
Upon execution, the script will generate visualizations as PNG files and compile a detailed PDF report named Sales_Analysis_Report.pdf.
</agent_response> """,
    f"""<thinking>To solve 'Create a JavaScript program that dynamically generates a to-do list with add, delete, and filter functionalities', I will outline the steps to set up the HTML structure, write the CSS for styling, and develop the JavaScript code to handle user interactions and data management.</thinking> <step>Design the HTML structure for the to-do list application.</step> <count>5</count> <thinking>The HTML will include an input field for adding new tasks, buttons for adding and filtering tasks, and a container to display the list of to-do items. Each to-do item will have a checkbox for completion and a delete button.</thinking> <step>Set up the basic HTML elements needed for the to-do list interface.</step> <count>4</count> <thinking>Style the application using CSS to make it visually appealing and user-friendly. This includes layout, colors, fonts, and responsive design considerations.</thinking> <step>Develop the CSS to style the to-do list application, ensuring a clean and intuitive user interface.</step> <count>3</count> <thinking>Implement JavaScript functions to handle adding new tasks, deleting existing ones, marking tasks as completed, and filtering tasks based on their status (all, completed, pending).</thinking> <step>Write JavaScript code to manage task creation, deletion, completion, and filtering functionalities.</step> <count>2</count> <thinking>Ensure that the application stores tasks locally using the browser's localStorage so that the to-do list persists across sessions.</thinking> <step>Incorporate localStorage to save and retrieve tasks, maintaining data persistence.</step> <count>1</count> <thinking>Test the application thoroughly to ensure all functionalities work as intended and handle edge cases, such as adding empty tasks or deleting tasks.</thinking> <step>Perform testing and debugging to validate the application's functionality and user experience.</step> <count>0</count> <agent_response>

html
Copy code
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dynamic To-Do List</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="todo-container">
        <h1>My To-Do List</h1>
        <div class="input-section">
            <input type="text" id="task-input" placeholder="Add a new task...">
            <button id="add-task-btn">Add Task</button>
        </div>
        <div class="filter-section">
            <button class="filter-btn" data-filter="all">All</button>
            <button class="filter-btn" data-filter="completed">Completed</button>
            <button class="filter-btn" data-filter="pending">Pending</button>
        </div>
        <ul id="task-list">
            <!-- Tasks will be dynamically added here -->
        </ul>
    </div>

    <script src="script.js"></script>
</body>
</html>
css
Copy code
/* styles.css */
body {{
    background-color: #f4f4f4;
    font-family: Arial, sans-serif;
}}

.todo-container {{
    width: 400px;
    margin: 50px auto;
    background: #fff;
    padding: 30px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}}

h1 {{
    text-align: center;
    color: #333;
}}

.input-section {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 20px;
}}

#task-input {{
    width: 75%;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
}}

#add-task-btn {{
    width: 20%;
    padding: 10px;
    background: #28a745;
    color: #fff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}}

#add-task-btn:hover {{
    background: #218838;
}}

.filter-section {{
    display: flex;
    justify-content: space-around;
    margin-bottom: 20px;
}}

.filter-btn {{
    padding: 10px 15px;
    border: none;
    background: #007bff;
    color: #fff;
    border-radius: 4px;
    cursor: pointer;
}}

.filter-btn:hover {{
    background: #0069d9;
}}

#task-list {{
    list-style-type: none;
    padding: 0;
}}

.task-item {{
    display: flex;
    justify-content: space-between;
    padding: 10px;
    border-bottom: 1px solid #ddd;
}}

.task-item.completed .task-text {{
    text-decoration: line-through;
    color: #888;
}}

.task-text {{
    flex-grow: 1;
    margin-left: 10px;
}}

.delete-btn {{
    background: #dc3545;
    border: none;
    color: #fff;
    padding: 5px 10px;
    border-radius: 4px;
    cursor: pointer;
}}

.delete-btn:hover {{
    background: #c82333;
}}
javascript
Copy code
// script.js

// Select DOM elements
const taskInput = document.getElementById('task-input');
const addTaskBtn = document.getElementById('add-task-btn');
const taskList = document.getElementById('task-list');
const filterButtons = document.querySelectorAll('.filter-btn');

// Load tasks from localStorage on page load
document.addEventListener('DOMContentLoaded', loadTasks);

// Add task event
addTaskBtn.addEventListener('click', addTask);

// Filter tasks event
filterButtons.forEach(button => {{
    button.addEventListener('click', () => {{
        filterTasks(button.dataset.filter);
    }});
}});

// Function to add a new task
function addTask() {{
    const taskText = taskInput.value.trim();
    if (taskText === '') {{
        alert('Please enter a task.');
        return;
    }}

    const task = {{
        id: Date.now(),
        text: taskText,
        completed: false
    }};

    saveTask(task);
    renderTask(task);
    taskInput.value = '';
}}

// Function to save task to localStorage
function saveTask(task) {{
    let tasks = getTasksFromStorage();
    tasks.push(task);
    localStorage.setItem('tasks', JSON.stringify(tasks));
}}

// Function to get tasks from localStorage
function getTasksFromStorage() {{
    let tasks;
    if (localStorage.getItem('tasks') === null) {{
        tasks = [];
    }} else {{
        tasks = JSON.parse(localStorage.getItem('tasks'));
    }}
    return tasks;
}}

// Function to load tasks from localStorage
function loadTasks() {{
    const tasks = getTasksFromStorage();
    tasks.forEach(task => renderTask(task));
}}

// Function to render a task in the DOM
function renderTask(task) {{
    const li = document.createElement('li');
    li.className = 'task-item';
    if (task.completed) {{
        li.classList.add('completed');
    }}
    li.setAttribute('data-id', task.id);

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = task.completed;
    checkbox.addEventListener('change', toggleTask);

    const span = document.createElement('span');
    span.className = 'task-text';
    span.textContent = task.text;

    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'delete-btn';
    deleteBtn.textContent = 'Delete';
    deleteBtn.addEventListener('click', deleteTask);

    li.appendChild(checkbox);
    li.appendChild(span);
    li.appendChild(deleteBtn);

    taskList.appendChild(li);
}}

// Function to toggle task completion
function toggleTask(e) {{
    const listItem = e.target.parentElement;
    const taskId = listItem.getAttribute('data-id');
    const tasks = getTasksFromStorage();
    tasks.forEach(task => {{
        if (task.id == taskId) {{
            task.completed = e.target.checked;
        }}
    }});
    localStorage.setItem('tasks', JSON.stringify(tasks));
    listItem.classList.toggle('completed');
}}

// Function to delete a task
function deleteTask(e) {{
    const listItem = e.target.parentElement;
    const taskId = listItem.getAttribute('data-id');
    listItem.remove();

    let tasks = getTasksFromStorage();
    tasks = tasks.filter(task => task.id != taskId);
    localStorage.setItem('tasks', JSON.stringify(tasks));
}}

// Function to filter tasks
function filterTasks(filter) {{
    const tasks = taskList.children;
    Array.from(tasks).forEach(task => {{
        switch(filter) {{
            case 'all':
                task.style.display = 'flex';
                break;
            case 'completed':
                if (task.classList.contains('completed')) {{
                    task.style.display = 'flex';
                }} else {{
                    task.style.display = 'none';
                }}
                break;
            case 'pending':
                if (!task.classList.contains('completed')) {{
                    task.style.display = 'flex';
                }} else {{
                    task.style.display = 'none';
                }}
                break;
        }}
    }});
}}
Explanation:

This JavaScript program creates a dynamic to-do list application with functionalities to add, delete, and filter tasks. The application also persists data using the browser's localStorage. Here's a detailed breakdown of the implementation:

HTML Structure (index.html):

Input Section: Contains an input field for entering new tasks and an "Add Task" button.
Filter Section: Includes buttons to filter tasks based on their status: All, Completed, and Pending.
Task List: An unordered list (<ul>) where tasks are dynamically added as list items (<li>).
Styling (styles.css):

Layout: Centers the to-do list container and styles the input, buttons, and task items for a clean and user-friendly interface.
Responsive Design: Ensures that the application looks good on various screen sizes.
Visual Indicators: Uses different styles to indicate completed tasks (e.g., strikethrough text).
JavaScript Functionality (script.js):

Task Management:
Adding Tasks: Users can add new tasks by entering text and clicking the "Add Task" button. The task is then saved to localStorage and rendered in the task list.
Deleting Tasks: Each task has a "Delete" button that allows users to remove it from the list and localStorage.
Marking as Completed: Users can mark tasks as completed by checking the corresponding checkbox. Completed tasks are visually distinguished and can be filtered accordingly.
Filtering Tasks:
Users can filter tasks to view all, only completed, or only pending tasks using the filter buttons. The filtering logic adjusts the display of tasks based on their completion status.
Data Persistence:
The application uses localStorage to save tasks, ensuring that the to-do list remains intact even after the browser is closed or refreshed.
Event Listeners:
The script sets up event listeners for adding tasks, toggling completion status, deleting tasks, and filtering tasks to ensure responsive and interactive user experiences.
Usage Instructions:

Setup:
Ensure that all three files (index.html, styles.css, and script.js) are placed in the same directory.
Running the Application:
Open index.html in a web browser.
Add tasks using the input field and "Add Task" button.
Mark tasks as completed using the checkboxes.
Delete tasks using the "Delete" buttons.
Filter tasks using the "All," "Completed," and "Pending" buttons.
Data Persistence:
Tasks are saved in the browser's localStorage, so they will persist across sessions.
Enhancements:

Validation: The script includes validation to prevent adding empty tasks.
User Feedback: Alerts notify users if they attempt to add an empty task.
Clean UI: The application provides a visually appealing and intuitive interface for managing tasks.
This comprehensive implementation ensures a functional and user-friendly to-do list application that effectively demonstrates dynamic DOM manipulation, event handling, and data persistence using JavaScript.

</agent_response> """,
]


class PrintSaver:
    def __init__(self):
        self.prints = {}
        self.line_count = 0
        self.save_filename = (
            f"advanced_prompting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    def print_and_store(self, str_to_save: str):
        print(str_to_save)
        # separate the prints by new lines
        for line in str_to_save.split("\n"):
            self.prints[self.line_count] = line
            self.line_count += 1

    def save_prints(self):
        self.prints = dict(sorted(self.prints.items(), key=lambda x: x[0]))
        with open(f"{self.save_filename}.txt", "w") as file:
            for line_num, line in self.prints.items():
                file.write(f"{line_num}: {line}\n")
        print(f"Prints saved to {self.save_filename}.txt")


print_saver = PrintSaver()


class PromptSuggestions(BaseModel):
    prompt_modifications: List[str] = Field(..., title="List of Prompt Suggestions")


class response_choices(BaseModel):
    number_of_selected_response: int = Field(..., title="Number of selected response")


class PromptEngineeringConfig:
    """Configuration settings for advanced prompt engineering techniques.

    This class encapsulates various parameters used in prompt engineering approaches,
    including multi-agent reasoning, self-consistency checks, and adaptive complexity handling.

    Attributes:
        max_steps (int): Maximum number of reasoning steps allowed (default: 20)
        initial_budget (int): Initial token/computation budget (default: 20)
        confidence_thresholds (Tuple[float, float, float]): Thresholds for high, medium,
            and low confidence levels (default: (
                0.8,  # High confidence threshold
                0.5,  # Medium confidence threshold
                0.0   # Low confidence threshold
            ))
        model (str): Name of the language model to use (default: "gpt-4o-mini")
        temperature (float): Sampling temperature for response generation (default: 0.7)
        top_p (float): Top-p (nucleus) sampling parameter (default: 0.9)
        n (int): Number of self-consistent samples to generate (default: 3)
        max_retries (int): Maximum number of retry attempts for failed generations (default: 3)
        backtrack (bool): Enable/disable backtracking in reasoning steps (default: True)
        agents (int): Number of agents for collaborative reasoning (default: 3)
        complexity_factor (int): Multiplier for adjusting step budget based on task
            complexity (default: 5)
    """

    max_steps: int = 20
    initial_budget: int = 20
    confidence_thresholds: Tuple[float, float, float] = (0.8, 0.5, 0.0)
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    top_p: float = 0.9
    n: int = 3  # Number of self-consistent samples
    max_retries: int = 3
    backtrack: bool = True
    agents: int = (
        3  # Number of agents for Collaborative Multi-Agent Reasoning, where multiple agents work together to solve a task by sharing insights and refining each other's suggestions.
    )
    complexity_factor: int = 5  # Factor to adjust step budget based on complexity

    def __init__(
        self,
        max_steps: int = 20,
        initial_budget: int = 20,
        confidence_thresholds: Tuple[float, float, float] = (0.8, 0.5, 0.0),
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 3,
        max_retries: int = 3,
        backtrack: bool = True,
        agents: int = 3,
        complexity_factor: int = 5,
    ):
        """Initialize the configuration settings.

        Args:
            max_steps: Maximum number of reasoning steps allowed
            initial_budget: Initial token/computation budget
            confidence_thresholds: Thresholds for high, medium, and low confidence
            model: Name of the language model to use
            temperature: Sampling temperature for response generation
            top_p: Top-p (nucleus) sampling parameter
            n: Number of self-consistent samples to generate
            max_retries: Maximum number of retry attempts
            backtrack: Enable/disable backtracking in reasoning
            agents: Number of agents for collaborative reasoning
            complexity_factor: Multiplier for adjusting step budget
        """
        self.max_steps = max_steps
        self.initial_budget = initial_budget
        self.confidence_thresholds = confidence_thresholds
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.max_retries = max_retries
        self.backtrack = backtrack
        self.agents = agents
        self.complexity_factor = complexity_factor


# CompnentType represents a category of different final output component types, ie. whether the output is its own standalone file, a part of a larger file, or a response to a prompt.

ComponentType = {
    "standalone_file": "standalone_file",
    "function_in_larger_file": "function_in_larger_file",
    "class_in_larger_file": "class_in_larger_file",
    "response_to_prompt": "response_to_prompt",
    "page_in_larger_file": "page_in_larger_file",
    "section_of_page": "section_of_page",
    "subsection_of_section": "subsection_of_section",
    "chapter_of_book": "chapter_of_book",
    "other": "other",
}
# ...existing code...


class StepComponentType(BaseModel):
    standalone: bool = Field(..., title="Whether the component is a standalone file")
    component_type: str = Field(
        ...,
        title="Component Type, should be one of the following: standalone_file, function_in_larger_file, class_in_larger_file, response_to_prompt, page, section, subsection, chapter",
    )


class Reflection(BaseModel):
    content: str = Field(
        ...,
        title="Content of the reflection (insights, learnings, corrections) or feedback regarding the previous step",
    )
    reward: float = Field(..., title="Reward value associated with the reflection")
    step_number: int = Field(..., title="Step number where the reflection occurred")

    def __repr__(self):
        return f"Reflection(content={self.content}, reward={self.reward}, step_number={self.step_number})"

    def __str__(self):
        return f"Reflection(content={self.content}, reward={self.reward}, step_number={self.step_number})"

    def __eq__(self, other):
        return (
            self.content == other.content
            and self.reward == other.reward
            and self.step_number == other.step_number
        )

    def __hash__(self):
        return hash((self.content, self.reward, self.step_number))


class Step(BaseModel):
    description: str = Field(..., title="Description of the step")
    step_number: int = Field(..., title="Sequential number of the step")
    remaining_budget: int = Field(..., title="Remaining budget at this step")
    reflection: Optional[Reflection] = Field(None, title="Reflection object")
    thoughts: Optional[str] = Field(None, title="Additional thoughts about the step")
    plan_step_number: Optional[int] = Field(
        None, title="Corresponding step number in the plan"
    )
    final_step_output: Optional[FinalStepOutput] = Field(
        None, title="Final output generated at this step"
    )
    # """
    # A class representing a step in a process.

    # Attributes:
    #     description (str): A text description of the step.
    #     step_number (int): The sequential number/position of this step.
    #     remaining_budget (int): The budget amount remaining at this step.
    #     reflection (Reflection): A reflection object containing insights and rewards.
    #     thoughts (str): Additional thoughts or considerations about the step.
    #     plan_step_number (int): The corresponding step number in the plan.
    #     final_step_output (str): The final output generated at this step.
    # """

    # def __init__(
    #     self,
    #     description: str,
    #     step_number: int,
    #     remaining_budget: int,
    #     reflection: Reflection | None = None,
    #     thoughts: str = None,
    #     plan_step_number: int = None,
    #     final_step_output: FinalStepOutput | None = None,
    # ):
    #     self.description = description
    #     self.step_number = step_number
    #     self.remaining_budget = remaining_budget
    #     self.reflection = reflection
    #     self.thoughts = thoughts
    #     self.plan_step_number = plan_step_number
    #     self.final_step_output = final_step_output

    def __repr__(self):
        return f"Step(description={self.description}, step_number={self.step_number}, remaining_budget={self.remaining_budget}, reflection={self.reflection})"

    def __str__(self):
        return f"Step(description={self.description}, step_number={self.step_number}, remaining_budget={self.remaining_budget}, reflection={self.reflection})"

    def __eq__(self, other):
        # Helper function for string comparison
        def fuzzy_string_match(str1, str2, threshold=0.75):

            if str1 is None and str2 is None:
                return True
            if str1 is None or str2 is None:
                return False

            # Normalize strings
            def normalize(s):
                s = s.lower().strip()
                s = re.sub(r"[^\w\s]", "", s)  # Remove punctuation
                s = " ".join(s.split())  # Normalize whitespace
                return s

            str1 = normalize(str1)
            str2 = normalize(str2)

            # Length-based filtering
            if abs(len(str1) - len(str2)) / max(len(str1), len(str2)) > 0.3:
                return False

            # Substring checks
            if str1 in str2 or str2 in str1 or str1 == str2:
                return True
            if len(str1) > 10 and str1[5:-5] in str2:
                return True
            if len(str2) > 10 and str2[5:-5] in str1:
                return True

            # Similarity ratio using difflib
            sequence_matcher_ratio = difflib.SequenceMatcher(None, str1, str2).ratio()
            if sequence_matcher_ratio >= threshold:
                return True

            # Levenshtein ratio check
            levenshtein_similarity = ratio(str1, str2)
            return levenshtein_similarity >= threshold

        return (
            fuzzy_string_match(
                self.description if self.description else None,
                other.description if other.description else None,
            )
            and self.step_number == other.step_number
            and self.remaining_budget == other.remaining_budget
            and fuzzy_string_match(
                self.reflection.content if self.reflection else None,
                other.reflection.content if other.reflection else None,
            )
        )

    def __hash__(self):
        return hash(
            (
                self.description,
                self.step_number,
                self.remaining_budget,
                self.reflection,
            )
        )


class FinalStepOutput(BaseModel):
    final_output: str = Field(
        ..., title="Final output", description="The final output string"
    )
    output_type: OutputType = Field(
        ..., title="Output type", description="The type of the output"
    )
    version: int = Field(
        ..., title="Version", description="Version number of the output"
    )
    component_type: str = Field(
        ...,
        title="Component type",
        description="The type of the component generating the output",
    )
    associated_plan_step: PlanStep = Field(
        ...,
        title="Associated Plan Step",
        description="The plan step associated with this output",
    )
    step: Step = Field(
        ..., title="Step", description="The step object associated with this output"
    )

    @field_validator("step", mode="before")
    def accept_step_instance(cls, value):
        # Accept if it's already an instance of Step
        if isinstance(value, Step):
            return value
        # If a dataclass (but not a Pydantic model), convert to dict and then validate
        elif is_dataclass(value):
            return Step.model_validate(asdict(value))
        elif isinstance(value, dict):
            return Step.model_validate(value)
        raise ValueError("step must be a Step instance, a dataclass, or dict")


class FinalPlanStepOutput(BaseModel):
    final_output: str = Field(..., title="Final output generated at this step")
    output_type: OutputType = Field(..., title="Type of output generated")
    version: int = Field(..., title="Version of the output")
    component_type: str = Field(..., title="Type of component")
    steps: List[Step] = Field(..., title="List of steps associated with this output")
    planstep: PlanStep = Field(..., title="PlanStep object associated with this output")
    file_name: str = Field(..., title="Name of the file where the output is stored")
    parent_file_name: str = Field(
        ..., title="Name of the parent file where the output is stored"
    )


class Task:
    """Task class represents a structured task with its description, complexity, and execution details.

    Attributes:
        description (str): The initial task description provided.
        refined_description (str): A clarified or improved version of the task description.
        complexity (int): A numerical value indicating the task's complexity level.
        steps (List[str]): Ordered list of steps required to complete the task.
        reflections (List[str]): Collection of insights or observations about the task.
        answer (str): The final solution or response to the task.
        final_reward (float): A numerical value representing the task completion reward or score.
        plan (Plan): An optional plan object containing detailed steps and subtasks.
    """

    description: str
    refined_description: str
    complexity: int
    steps: List[Step]
    reflections: List[Reflection]
    answer: str
    final_reward: float
    plan = Plan
    output_type: OutputType
    planstep_outputs: List[FinalPlanStepOutput]

    def __init__(
        self,
        description: str,
        refined_description: str,
        complexity: int,
        steps: List[Step],
        reflections: List[Reflection],
        answer: str,
        final_reward: float,
        plan: Plan,
        output_type: OutputType,
        project_name: str = None,
    ):
        self.description = description
        self.refined_description = refined_description
        self.complexity = complexity
        self.steps = steps
        self.reflections = reflections
        self.answer = answer
        self.final_reward = final_reward
        self.plan = plan
        self.output_type = output_type
        self.project_name = project_name
        self.planstep_outputs = []

    def __repr__(self):
        return f"Task(description={self.description}, complexity={self.complexity}, steps={self.steps}, reflections={self.reflections}, answer={self.answer}, final_reward={self.final_reward}, plan={self.plan})"

    def __str__(self):
        return f"Task(description={self.description}, complexity={self.complexity}, steps={self.steps}, reflections={self.reflections}, answer={self.answer}, final_reward={self.final_reward}, plan={self.plan})"

    def __eq__(self, other):
        return (
            self.description == other.description
            and self.complexity == other.complexity
            and self.steps == other.steps
            and self.reflections == other.reflections
            and self.answer == other.answer
            and self.final_reward == other.final_reward
            and self.plan == other.plan
        )

    def __hash__(self):
        return hash(
            (
                self.description,
                self.complexity,
                tuple(self.steps),
                tuple(self.reflections),
                self.answer,
                self.final_reward,
                self.plan,
            )
        )


class Interaction:
    """A class representing an interaction session with steps, reflections, and outcomes.

    This class encapsulates the complete interaction flow including individual steps,
    reflections made during the process, final answer and reward.

    Attributes:
        task (Task): The task object associated with this interaction.
        steps (List[Step]): List of steps taken during the interaction.
        reflections (List[Reflection]): List of reflections made during the interaction.
        answer (Optional[str]): The final answer or response for the interaction, if any.
        final_reward (Optional[float]): The final numerical reward/score for the interaction, if any.
    """

    def __init__(
        self,
        task: Task,
        steps: List[Step],
        reflections: List[Reflection],
        answer: str = None,
        final_reward: float = 0.0,
        step_outputs: List[FinalStepOutput] = [],
        planstep_outputs: List[FinalPlanStepOutput] = [],
        existing_files: List[str] = [],
    ):
        self.task = task
        self.steps = steps
        self.reflections = reflections
        self.answer = answer
        self.final_reward = final_reward
        self.step_outputs = step_outputs
        self.planstep_outputs = planstep_outputs
        self.existing_files = existing_files

    def __repr__(self):
        return f"Interaction(task={self.task}, steps={self.steps}, reflections={self.reflections}, answer={self.answer}, final_reward={self.final_reward})"

    def __str__(self):
        return f"Interaction(task={self.task}, steps={self.steps}, reflections={self.reflections}, answer={self.answer}, final_reward={self.final_reward})"

    def __eq__(self, other):
        return (
            self.task == other.task
            and self.steps == other.steps
            and self.reflections == other.reflections
            and self.answer == other.answer
            and self.final_reward == other.final_reward
        )

    def __hash__(self):
        return hash(
            (
                self.task,
                tuple(self.steps),
                tuple(self.reflections),
                self.answer,
                self.final_reward,
            )
        )

    def to_dict(self):
        return {
            "task": self.task,
            "steps": self.steps,
            "reflections": self.reflections,
            "answer": self.answer,
            "final_reward": self.final_reward,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            task=data["task"],
            steps=data["steps"],
            reflections=data["reflections"],
            answer=data["answer"],
            final_reward=data["final_reward"],
        )

    def sum_rewards(self):
        total_reward = sum(reflection.reward for reflection in self.reflections)
        print_saver.print_and_store(f"Total Reward: {total_reward}")
        return total_reward

    def save_final_outputs_to_logs(self):
        """Save the final outputs of the interaction to a text file."""
        # Create project directory if it doesn't exist
        os.makedirs(self.task.project_name, exist_ok=True)

        with open(
            os.path.join(self.task.project_name, "final_outputs.txt"), "w"
        ) as file:
            file.write("Final Outputs:\n")
            for planstep_output in self.planstep_outputs:
                file.write(
                    f"Plan Step Number: {planstep_output.planstep.step_number}\n"
                )
                file.write(f"Final Output: {planstep_output.final_output}\n")
                for step_output in planstep_output.steps:
                    if (
                        step_output.step.plan_step_number
                        == planstep_output.planstep.step_number
                    ):
                        file.write(f"Step Number: {step_output.step.step_number}\n")
                        file.write(
                            f"Final Output: {step_output.final_output if step_output.final_output else step_output.step.description}\n"
                        )
                file.write("\n")

    def save_final_file(self):
        """Save the final output of the interaction to a text file."""
        file_extension = self.task.output_type.file_extension
        filenames_list = self.existing_files

        # Create project directory if it doesn't exist
        os.makedirs(self.task.project_name, exist_ok=True)

        for planstep in self.planstep_outputs:
            if planstep.component_type == ComponentType["standalone_file"]:
                filename = os.path.join(
                    self.task.project_name,
                    f"{planstep.file_name or f'output_{planstep.planstep.step_number}'}{file_extension}",
                )
                filenames_list.append(planstep.file_name)
                try:
                    with open(filename, "a") as file:
                        file.write(planstep.final_output)
                except IOError as e:
                    raise IOError(f"Failed to write to {filename}: {e}")

            elif planstep.component_type in [
                ComponentType["function_in_larger_file"],
                ComponentType["class_in_larger_file"],
            ]:
                filename = (
                    os.path.join(
                        self.task.project_name,
                        f"{planstep.parent_file_name}{file_extension}",
                    )
                    if planstep.parent_file_name
                    else filenames_list[-1]
                )

                if (
                    planstep.parent_file_name
                    and planstep.parent_file_name not in filenames_list
                ):
                    filenames_list.append(planstep.parent_file_name)
                try:
                    with open(filename, "a") as file:
                        file.write(planstep.final_output)
                except IOError as e:
                    raise IOError(f"Failed to write to {filename}: {e}")
        self.existing_files = filenames_list


class CompletionStatus(BaseModel):
    completion: bool = Field(..., title="Completion Status")


class FinalReviewScore(BaseModel):
    score: float


class StepNumber(BaseModel):
    score: int


class AdvancedPromptEngineer:
    def __init__(self, config: PromptEngineeringConfig):
        self.config = config
        self.knowledge_base = {}  # For Retrieval-Augmented Generation
        self.task_object = None

    def count_tokens(self, text: list[str]) -> int:
        """Count the number of tokens in the input text using the tiktoken library."""
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        total_tokens = 0
        for text_item in text:
            total_tokens += len(encoding.encode(text_item))
        return total_tokens

    def track_progress(self, interaction: Interaction) -> None:
        """Track progress using tqdm progress bar"""

        completed_steps = len(interaction.steps)
        total_steps = self.config.initial_budget

        # Create progress bar
        with tqdm(total=total_steps, desc="Progress") as pbar:
            pbar.n = completed_steps
            pbar.refresh()

        progress = (completed_steps / total_steps) * 100
        self.logger.info(f"Progress: {progress:.2f}%")

    def name_file(self, file_string: str, file_extension: str) -> str:
        """Name the file based on the content of the file."""

        system_prompt = f"You are tasked with discerning the intended name of a file based on its content. The file contains the following text: '{file_string}'. Please provide a suitable name for this file. If there does not seem to be an intended filename, you may suggest a descriptive filename based on the content."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": system_prompt},
        ]
        try:
            prompt_response = openai.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.2,
            )

            return prompt_response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def name_project(self, task: Task, plan: Plan) -> str:
        """Name the project based on everything."""

        system_prompt = f"You are tasked with discerning the intended name of a project based on its content. The project contains the following text: '{task.description}'. Please provide a suitable name for this project. If there does not seem to be an intended project name, you may suggest a descriptive project name based on the content. The plan for the project is as follows: {plan}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": system_prompt},
        ]
        try:
            prompt_response = openai.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.2,
            )

            return prompt_response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def convert_planstep_to_prompt(self, plan_step: PlanStep, task: Task) -> str:
        """Convert a PlanStep object to a prompt for the language model."""

        planstep_outputs = [
            (
                (task.planstep_outputs[output_index].final_output, True)
                if output_index < len(task.planstep_outputs)
                else ("No output yet", False)
            )
            for output_index in range(len(task.plan.steps))
        ]
        plan_steps_strs = [
            {
                f"""\n Plan Step Number: {plan_step.step_number} \n
            {plan_step.step_name}: {plan_step.step_description} \n
            Plan Step Explanation: {plan_step.step_explanation} \n
            Plan Step Expected Output: {plan_step.step_output} \n
            Plan Step Full Text: {plan_step.step_full_text} \n
            Plan Step Completed? {plan_step.completed} \n
            Plan Step Output: {output} """: plan_step.step_number
            }
            for plan_step, output in zip(task.planstep_outputs, planstep_outputs)
        ]

        plan_steps_strs = sorted(plan_steps_strs, key=lambda x: x.values())

        full_plan = "\n".join(
            [list(plan_step.keys())[0] for plan_step in plan_steps_strs]
        )

        example_conversion = f"""
        Please write a prompt for the following Plan Step:
        Plan Step Number: 2
        Plan Step Name: 'Write and Test the Function'
        Plan Step Description: 'Develop the core logic of the Python function to calculate the area of a circle, ensuring it meets the defined requirements. Additionally, create tests to verify that each part of the function operates correctly under various conditions.'
        Plan Step Explanation: 'In this step, focus on implementing the mathematical formula for the area of a circle within a function. Ensure that the function accepts user input for the radius, handles invalid inputs gracefully, and returns the correct area. After writing the function, design and execute test cases to validate its accuracy and robustness, including edge cases such as zero or negative inputs.'
        Plan Step Expected Output: 'A Python function named `calculate_circle_area` that accurately computes the area based on user input, along with a set of test cases that confirm the function works as intended.'
        Plan Step Full Text: 'Implement a Python function `calculate_circle_area(radius)` that calculates the area using the formula πr². The function should prompt the user to enter the radius, validate the input to ensure it is a positive number, and return the calculated area. Following the implementation, write test cases to verify the function’s correctness, including tests for typical values, zero, and negative inputs to ensure proper error handling.'

        This Plan Step is part of the following Task:
        Task Description: 'Create a Python application that interactively calculates the area of a circle based on user input. The application should follow a structured development plan, emphasizing incremental development, testing, and user input validation to ensure reliability and accuracy.'

        The overall Plan is as follows:

        Plan Step Number: 1
        Plan Step Name: 'Define the Purpose and Requirements'
        Plan Step Description: 'Outline the main objectives of the Python application, specifying the inputs, outputs, and functionalities it should possess.'
        Plan Step Explanation: 'Determine what the application aims to achieve, including how it interacts with the user, what calculations it performs, and any constraints or specifications it must adhere to.'
        Plan Step Expected Output: 'A clear and concise set of requirements that define the scope and functionality of the Python application.'
        Plan Step Full Text: 'Identify the key features of the application, such as accepting user input for the radius, calculating the area of a circle using the formula πr², displaying the result, and handling invalid inputs gracefully.'
        Is Plan Step Completed? Yes
        Plan Step Final Output: 
        'Objective of Application: Calculate the area of a circle based on user input of the radius.
        Requirements: Accept user input for the radius, validate the input, calculate the area using the formula πr², display the result, handle invalid inputs. 
        Constraints: Ensure the radius is a positive number, provide clear user instructions, test the application thoroughly.
        Notes: The application should be user-friendly, robust, and well-tested.'

        Plan Step Number: 2
        Plan Step Name: 'Write and Test the Function'
        Plan Step Description: 'Develop the core logic of the Python function to calculate the area of a circle, ensuring it meets the defined requirements. Additionally, create tests to verify that each part of the function operates correctly under various conditions.'
        Plan Step Explanation: 'In this step, focus on implementing the mathematical formula for the area of a circle within a function. Ensure that the function accepts user input for the radius, handles invalid inputs gracefully, and returns the correct area. After writing the function, design and execute test cases to validate its accuracy and robustness, including edge cases such as zero or negative inputs.'
        Plan Step Expected Output: 'A Python function named `calculate_circle_area` that accurately computes the area based on user input, along with a set of test cases that confirm the function works as intended.'
        Plan Step Full Text: 'Implement a Python function `calculate_circle_area(radius)` that calculates the area using the formula πr². The function should prompt the user to enter the radius, validate the input to ensure it is a positive number, and return the calculated area. Following the implementation, write test cases to verify the function’s correctness, including tests for typical values, zero, and negative inputs to ensure proper error handling.'
        Is Plan Step Completed? No
        Plan Step Final Output: None

        Plan Step Number: 3
        Plan Step Name: 'Create the User Interface'
        Plan Step Description: 'Design a simple command-line interface that interacts with the user, collects input, and displays the calculated area.'
        Plan Step Explanation: 'Develop a user-friendly interface that prompts the user to enter the radius of the circle. Ensure that the interface handles user inputs effectively, displays meaningful messages, and integrates seamlessly with the `calculate_circle_area` function.'
        Plan Step Expected Output: 'A functional command-line interface that interacts with the user to receive input and display the calculated area.'
        Plan Step Full Text: 'Design and implement a command-line interface that prompts the user to input the radius of a circle. The interface should call the `calculate_circle_area` function with the provided input and display the resulting area. Additionally, handle invalid inputs by informing the user of the error and prompting them to enter a valid radius.'
        Is Plan Step Completed? No
        Plan Step Final Output: None

        Plan Step Number: 4
        Plan Step Name: 'Enhance and Document the Application'
        Plan Step Description: 'Improve the application by adding features such as input validation, error handling, and comprehensive documentation.'
        Plan Step Explanation: 'Refine the existing code to make it more robust and user-friendly. Implement thorough input validation to prevent incorrect data entries, add error handling to manage unexpected scenarios gracefully, and document the code to explain its functionality and usage.'
        Plan Step Expected Output: 'An enhanced Python application with improved reliability, user experience, and well-documented code.'
        Plan Step Full Text: 'Enhance the application by adding input validation to ensure that the radius entered is a positive number. Implement error handling to manage scenarios such as non-numeric inputs or other unexpected user behaviors. Additionally, write comprehensive documentation and comments within the code to explain the purpose of each function, the flow of the application, and instructions for users on how to operate the program.'
        Is Plan Step Completed? No
        Plan Step Final Output: None

        As a reminder, you are writing a prompt to guide an LLM to complete the following plan step:
        Plan Step Number: 2
        Plan Step Name: 'Write and Test the Function'
        Plan Step Description: 'Develop the core logic of the Python function to calculate the area of a circle, ensuring it meets the defined requirements. Additionally, create tests to verify that each part of the function operates correctly under various conditions.'
        Plan Step Explanation: 'In this step, focus on implementing the mathematical formula for the area of a circle within a function. Ensure that the function accepts user input for the radius, handles invalid inputs gracefully, and returns the correct area. After writing the function, design and execute test cases to validate its accuracy and robustness, including edge cases such as zero or negative inputs.'
        Plan Step Expected Output: 'A Python function named `calculate_circle_area` that accurately computes the area based on user input, along with a set of test cases that confirm the function works as intended.'
        Is Plan Step Completed? No
        Plan Step Final Output: None

        Prompt:
        """
        faux_response = f"Your objective is to develop and test a Python function named `calculate_circle_area` that computes the area of a circle based on a user-provided radius. Start by implementing the core logic using the formula πr², ensuring that the function accepts input, validates that the radius is a positive number, and returns the correct area. After writing the function, create a series of test cases to verify its accuracy, including typical radius values, zero, and negative numbers to test input validation and error handling. Focus on building a reliable and well-tested function without providing detailed step-by-step instructions or the final code solution."

        init_prompt = f"""
        You are an expert prompt engineer. Your job is to write prompts for the next step based on the provided Plan Step. You write the prompt in such a way that it guides the user to complete the step effectively within the context of the overall Plan and the overarching Task, without providing the exact solution or providing unnecessary information.

        The prompt should follow these rules:
        1. Clearly explain the objective of the step.
        2. Provide any relevant context or information needed to complete the step. Use the rest of the Plan and Task as reference.
        3. AVOID attempting to provide the solution or performing the step yourself. Do not provide step-by-step instructions.
        4. Keep the prompt concise and focused on the step's objective, ensuring it aligns with the Plan Step and Task requirements.
        5. Do not include any information that is not directly related to the step or that may lead the LLM to do more than required or progress beyond the current step.
        6. Always assume the LLM is only aware of the one single step they are doing and nothing else, not of the overall plan or task.
        
        """

        this_step = f"""Please write a prompt for the following Plan Step:
        Plan Step Name: '{plan_step.step_name}'
        Plan Step Description: '{plan_step.step_description}'
        Plan Step Explanation: '{plan_step.step_explanation}'
        Plan Step Expected Output: '{plan_step.step_output}'
        Plan Step Full Text: '{plan_step.step_full_text}'

        This Plan Step is part of the following Task:
        Task Description: '{task.description}'
        
        The overall Plan is as follows:

        {full_plan}

        As a reminder, you are writing a prompt to guide an LLM to complete the following plan step:
        Plan Step Name: '{plan_step.step_name}'
        Plan Step Description: '{plan_step.step_description}'
        Plan Step Explanation: '{plan_step.step_explanation}'
        Plan Step Expected Output: '{plan_step.step_output}'

        Please respond only with the prompt text, and remember that the LLM does not remember previous steps or know anything beyond the current step.
        
        Prompt:
        """

        messages = [
            {"role": "system", "content": init_prompt},
            {"role": "user", "content": example_conversion},
            {"role": "assistant", "content": faux_response},
            {"role": "user", "content": this_step},
        ]
        try:
            prompt_response = openai.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.2,
            )
        except Exception as e:
            print_saver.print_and_store(f"Error: {e}")
            return f"Please write a prompt for the following Plan Step: {plan_step.step_name} - {plan_step.step_description}. The expected output is: {plan_step.step_output}. The full text of the Plan Step is: {plan_step.step_full_text} and the explanation is: {plan_step.step_explanation}. Your response should constitute a final output that meets the requirements."

        if prompt_response.choices[0].message.content:
            return str(prompt_response.choices[0].message.content)
        else:
            print_saver.print_and_store("Prompt generation failed.")
            return f"Please complete the following Plan Step: {plan_step.step_name} - {plan_step.step_description}. The expected output is: {plan_step.step_output}. The full text of the Plan Step is: {plan_step.step_full_text} and the explanation is: {plan_step.step_explanation}. Your response should constitute a final output that meets the requirements."

    def component_decision(self, task: Task, plan_step: PlanStep) -> str:
        """Decide the component type for the final output based on the task and plan step requirements."""

        planstep_outputs = [
            (
                (task.planstep_outputs[output_index].final_output, True)
                if output_index < len(task.planstep_outputs)
                else ("No output yet", False)
            )
            for output_index in range(len(task.plan.steps))
        ]
        plan_steps_strs = [
            {
                f"""\n Plan Step Number: {plan_step.step_number} \n
            {plan_step.step_name}: {plan_step.step_description} \n
            Plan Step Explanation: {plan_step.step_explanation} \n
            Plan Step Expected Output: {plan_step.step_output} \n
            Plan Step Full Text: {plan_step.step_full_text} \n
            Plan Step Completed? {plan_step.completed} \n
            Plan Step Output: {output} """: plan_step.step_number
            }
            for plan_step, output in zip(task.planstep_outputs, planstep_outputs)
        ]
        plan_steps_strs = sorted(plan_steps_strs, key=lambda x: x.values())

        full_plan = "\n".join(
            [list(plan_step.keys())[0] for plan_step in plan_steps_strs]
        )

        system_prompt = {
            "role": "system",
            "content": f"""You are an expert AI assistant tasked with determining the component type for the final output based on the task and plan step requirements. Your goal is to identify the most appropriate component type that aligns with the task's objectives and the plan step's expectations. Consider the nature of the output, such as whether it should be a standalone file, a function in a larger file, a class in a larger file, or a response to a prompt. Choose the component type that best suits the context and purpose of the final output.""",
        }

        example_decision = f"""
        Follows is an example of a component decision based on the task and plan step requirements:
        Please decide on the component type for the following Plan Step:
        Plan Step Number: 2
        Plan Step Name: 'Write and Test the Function'
        Plan Step Description: 'Develop the core logic of the Python function to calculate the area of a circle, ensuring it meets the defined requirements. Additionally, create tests to verify that each part of the function operates correctly under various conditions.'
        Plan Step Explanation: 'In this step, focus on implementing the mathematical formula for the area of a circle within a function. Ensure that the function accepts user input for the radius, handles invalid inputs gracefully, and returns the correct area. After writing the function, design and execute test cases to validate its accuracy and robustness, including edge cases such as zero or negative inputs.'
        Plan Step Expected Output: 'A Python function named `calculate_circle_area` that accurately computes the area based on user input, along with a set of test cases that confirm the function works as intended.'
        Plan Step Full Text: 'Implement a Python function `calculate_circle_area(radius)` that calculates the area using the formula πr². The function should prompt the user to enter the radius, validate the input to ensure it is a positive number, and return the calculated area. Following the implementation, write test cases to verify the function’s correctness, including tests for typical values, zero, and negative inputs to ensure proper error handling.'
        Is Plan Step Completed? Yes
        Final Plan Step Final Output: 

        This Plan Step is part of the following Task:
        Task Description: 'Create a Python application that interactively calculates the area of a circle based on user input. The application should follow a structured development plan, emphasizing incremental development, testing, and user input validation to ensure reliability and accuracy.'

        The overall Plan is as follows:

        The overall Plan is as follows:

        Plan Step Number: 1
        Plan Step Name: 'Define the Purpose and Requirements'
        Plan Step Description: 'Outline the main objectives of the Python application, specifying the inputs, outputs, and functionalities it should possess.'
        Plan Step Explanation: 'Determine what the application aims to achieve, including how it interacts with the user, what calculations it performs, and any constraints or specifications it must adhere to.'
        Plan Step Expected Output: 'A clear and concise set of requirements that define the scope and functionality of the Python application.'
        Plan Step Full Text: 'Identify the key features of the application, such as accepting user input for the radius, calculating the area of a circle using the formula πr², displaying the result, and handling invalid inputs gracefully.'
        Is Plan Step Completed? Yes
        Plan Step Final Output: 
        'Objective of Application: Calculate the area of a circle based on user input of the radius.
        Requirements: Accept user input for the radius, validate the input, calculate the area using the formula πr², display the result, handle invalid inputs. 
        Constraints: Ensure the radius is a positive number, provide clear user instructions, test the application thoroughly.
        Notes: The application should be user-friendly, robust, and well-tested.'

        Plan Step Number: 2
        Plan Step Name: 'Write and Test the Function'
        Plan Step Description: 'Develop the core logic of the Python function to calculate the area of a circle, ensuring it meets the defined requirements. Additionally, create tests to verify that each part of the function operates correctly under various conditions.'
        Plan Step Explanation: 'In this step, focus on implementing the mathematical formula for the area of a circle within a function. Ensure that the function accepts user input for the radius, handles invalid inputs gracefully, and returns the correct area. After writing the function, design and execute test cases to validate its accuracy and robustness, including edge cases such as zero or negative inputs.'
        Plan Step Expected Output: 'A Python function named `calculate_circle_area` that accurately computes the area based on user input, along with a set of test cases that confirm the function works as intended.'
        Plan Step Full Text: 'Implement a Python function `calculate_circle_area(radius)` that calculates the area using the formula πr². The function should prompt the user to enter the radius, validate the input to ensure it is a positive number, and return the calculated area. Following the implementation, write test cases to verify the function’s correctness, including tests for typical values, zero, and negative inputs to ensure proper error handling.'
        Is Plan Step Completed? No
        Plan Step Final Output: None

        Plan Step Number: 3
        Plan Step Name: 'Create the User Interface'
        Plan Step Description: 'Design a simple command-line interface that interacts with the user, collects input, and displays the calculated area.'
        Plan Step Explanation: 'Develop a user-friendly interface that prompts the user to enter the radius of the circle. Ensure that the interface handles user inputs effectively, displays meaningful messages, and integrates seamlessly with the `calculate_circle_area` function.'
        Plan Step Expected Output: 'A functional command-line interface that interacts with the user to receive input and display the calculated area.'
        Plan Step Full Text: 'Design and implement a command-line interface that prompts the user to input the radius of a circle. The interface should call the `calculate_circle_area` function with the provided input and display the resulting area. Additionally, handle invalid inputs by informing the user of the error and prompting them to enter a valid radius.'
        Is Plan Step Completed? No
        Plan Step Final Output: None

        Plan Step Number: 4
        Plan Step Name: 'Enhance and Document the Application'
        Plan Step Description: 'Improve the application by adding features such as input validation, error handling, and comprehensive documentation.'
        Plan Step Explanation: 'Refine the existing code to make it more robust and user-friendly. Implement thorough input validation to prevent incorrect data entries, add error handling to manage unexpected scenarios gracefully, and document the code to explain its functionality and usage.'
        Plan Step Expected Output: 'An enhanced Python application with improved reliability, user experience, and well-documented code.'
        Plan Step Full Text: 'Enhance the application by adding input validation to ensure that the radius entered is a positive number. Implement error handling to manage scenarios such as non-numeric inputs or other unexpected user behaviors. Additionally, write comprehensive documentation and comments within the code to explain the purpose of each function, the flow of the application, and instructions for users on how to operate the program.'
        Is Plan Step Completed? No
        Plan Step Final Output: None

        As a reminder, you are deciding on the component type of the output for the following Plan Step:
        Plan Step Number: 2
        Plan Step Name: 'Write and Test the Function'
        Plan Step Description: 'Develop the core logic of the Python function to calculate the area of a circle, ensuring it meets the defined requirements. Additionally, create tests to verify that each part of the function operates correctly under various conditions.'
        Plan Step Expected Output: 'A Python function named `calculate_circle_area` that accurately computes the area based on user input, along with a set of test cases that confirm the function works as intended.'

        The overall Task is as follows:
        Task Description: 'Create a Python application that interactively calculates the area of a circle based on user input. The application should follow a structured development plan, emphasizing incremental development, testing, and user input validation to ensure reliability and accuracy.'

        Please decide on the component type for the following Plan Step:
        Plan Step Number: 2
        Plan Step Name: 'Write and Test the Function'
        Plan Step Description: 'Develop the core logic of the Python function to calculate the area of a circle, ensuring it meets the defined requirements. Additionally, create tests to verify that each part of the function operates correctly under various conditions.'
        Plan Step Expected Output: 'A Python function named `calculate_circle_area` that accurately computes the area based on user input, along with a set of test cases that confirm the function works as intended.'

        Remember, reply with one of the following component types: 'standalone file', 'function in larger file', 'class in larger file', 'response to prompt', 'page', 'other'.
        Decision:\n
        """

        faux_response = f"function_in_larger_file"

        messages = [
            system_prompt,
            {"role": "system", "content": example_decision},
            {"role": "assistant", "content": faux_response},
        ]

        user_prompt = f"""
        Please decide on the component type for the following Plan Step:
        Plan Step Name: '{plan_step.step_name}'
        Plan Step Description: '{plan_step.step_description}'
        Plan Step Expected Output: '{plan_step.step_output}'
        Plan Step Full Text: '{plan_step.step_full_text}'
        Plan Step Explanation: '{plan_step.step_explanation}'
        Is Plan Step Completed? {plan_step.completed}
        Plan Step Output: {planstep_outputs[plan_step.step_number - 1][0]}

        This Plan Step is part of the following Task:
        Task Description: '{task.description}'

        The overall Plan is as follows:

        {full_plan}

        As a reminder, you are deciding on the component type of the output for the following Plan Step:
        Plan Step Name: '{plan_step.step_name}'
        Plan Step Description: '{plan_step.step_description}'
        Plan Step Expected Output: '{plan_step.step_output}'
        Plan Step Full Text: '{plan_step.step_full_text}'
        Plan Step Explanation: '{plan_step.step_explanation}'
        
        Remember, reply with one of the following component types: 'standalone file', 'function in larger file', 'class in larger file', 'response to prompt', 'page', 'other'.

        Decision:
        """
        messages.append({"role": "user", "content": user_prompt})
        try:
            decision_response = openai.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.2,
            )
        except Exception as e:
            print_saver.print_and_store(f"Error: {e}")
            return ComponentType["standalone_file"]

        if decision_response.choices[0].message.content:
            try:
                return (
                    ComponentType[decision_response.choices[0].message.content.strip()]
                    if decision_response.choices[0].message.content.strip()
                    in ComponentType.keys()
                    else ComponentType["standalone_file"]
                )
            except ValueError:
                return ComponentType["standalone_file"]

    def finalize_step_output(
        self, step: Step, task: Task, plan_step: PlanStep, previous_steps: List[Step]
    ) -> FinalStepOutput:
        """Finalize the output of a step based on the completion status and the task requirements. This function generates the final output based on the step's completion status and the task requirements. It does this by prompting an LLM to evaluate the step and then generating a response based on the evaluation.
        The LLM will be prompted to only make changes to the step if it is incomplete or incorrect, ensuring that the final output aligns with the task's objectives and the plan step's requirements.
        The LLM is allowed to 'clean up' the step, making minor modifications or improvements to enhance its quality and completeness, but it should not deviate significantly from the original step or the plan step's expectations.
        The LLMs output will represent the final version of the step, incorporating any necessary changes or additions to ensure its completion and alignment with the task and plan step.


        Args:
            step (Step): The step object.
            task (Task): The task object.
            plan_step (PlanStep): The corresponding plan step.

        Returns:
            FinalStepOutput: The finalized output of the step.
        """
        planstep_outputs = [
            (
                (task.planstep_outputs[output_index].final_output, True)
                if output_index < len(task.planstep_outputs)
                else ("No output yet", False)
            )
            for output_index in range(len(task.plan.steps))
        ]
        plan_steps_strs = [
            {
                f"""\n Plan Step Number: {plan_step.step_number} \n
            {plan_step.step_name}: {plan_step.step_description} \n
            Plan Step Explanation: {plan_step.step_explanation} \n
            Plan Step Expected Output: {plan_step.step_output} \n
            Plan Step Full Text: {plan_step.step_full_text} \n
            Plan Step Completed? {plan_step.completed} \n
            Plan Step Output: {output} """: plan_step.step_number
            }
            for plan_step, output in zip(task.planstep_outputs, planstep_outputs)
        ]
        plan_steps_strs = sorted(plan_steps_strs, key=lambda x: x.values())

        prev_step_strs = [
            f"<count>{stp_.remaining_budget +1}</count>\n<thinking>{stp_.thoughts}</thinking>\n<step>{stp_.description}</step>\n<reflection>{stp_.reflection}</reflection>\n"
            for stp_ in previous_steps
        ]

        prev_step_str = "".join(prev_step_strs)

        system_prompt = {
            "role": "system",
            "content": f"""You are an expert AI assistant tasked with finalizing the final output of a step. Your goal is to transcribe the step into the expected format that aligns with the task's requirements and the plan step's expectations. If the step is incomplete or incorrect, make the necessary modifications to ensure it fulfills the intended action described in the plan step. You can also make minor improvements to enhance the step's quality and completeness, but DO NOT make unnecessary changes or remove anything that isn't clearly an error or mistake. Your response should be a refined, final completed version of the step that meets the task's objectives and the plan step's requirements.""",
        }

        example_finalization = f"""
        Follows is an example of a finalized step output based on the completion status and the task requirements:
        Please finalize the following Step based on the Plan Step:
        Plan Step Number: 2
        Plan Step Name: 'Write and Test the Function'
        Plan Step Description: 'Develop the core logic of the Python function to calculate the area of a circle, ensuring it meets the defined requirements. Additionally, create tests to verify that each part of the function operates correctly under various conditions.'
        Plan Step Explanation: 'In this step, focus on implementing the mathematical formula for the area of a circle within a function. Ensure that the function accepts user input for the radius, handles invalid inputs gracefully, and returns the correct area. After writing the function, design and execute test cases to validate its accuracy and robustness, including edge cases such as zero or negative inputs.'
        Plan Step Expected Output: 'A Python function named `calculate_circle_area` that accurately computes the area based on user input, along with a set of test cases that confirm the function works as intended.'
        Plan Step Full Text: 'Implement a Python function `calculate_circle_area(radius)` that calculates the area using the formula πr². The function should prompt the user to enter the radius, validate the input to ensure it is a positive number, and return the calculated area. Following the implementation, write test cases to verify the function’s correctness, including tests for typical values, zero, and negative inputs to ensure proper error handling.'

        This Plan Step is part of the following Task:
        Task Description: 'Create a Python application that interactively calculates the area of a circle based on user input. The application should follow a structured development plan, emphasizing incremental development, testing, and user input validation to ensure reliability and accuracy.'

        The overall Plan is as follows:

        The overall Plan is as follows:

        Plan Step Number: 1
        Plan Step Name: 'Define the Purpose and Requirements'
        Plan Step Description: 'Outline the main objectives of the Python application, specifying the inputs, outputs, and functionalities it should possess.'
        Plan Step Explanation: 'Determine what the application aims to achieve, including how it interacts with the user, what calculations it performs, and any constraints or specifications it must adhere to.'
        Plan Step Expected Output: 'A clear and concise set of requirements that define the scope and functionality of the Python application.'
        Plan Step Full Text: 'Identify the key features of the application, such as accepting user input for the radius, calculating the area of a circle using the formula πr², displaying the result, and handling invalid inputs gracefully.'
        Is Plan Step Completed? Yes
        Plan Step Final Output: 
        'Objective of Application: Calculate the area of a circle based on user input of the radius.
        Requirements: Accept user input for the radius, validate the input, calculate the area using the formula πr², display the result, handle invalid inputs. 
        Constraints: Ensure the radius is a positive number, provide clear user instructions, test the application thoroughly.
        Notes: The application should be user-friendly, robust, and well-tested.'

        Plan Step Number: 2
        Plan Step Name: 'Write and Test the Function'
        Plan Step Description: 'Develop the core logic of the Python function to calculate the area of a circle, ensuring it meets the defined requirements. Additionally, create tests to verify that each part of the function operates correctly under various conditions.'
        Plan Step Explanation: 'In this step, focus on implementing the mathematical formula for the area of a circle within a function. Ensure that the function accepts user input for the radius, handles invalid inputs gracefully, and returns the correct area. After writing the function, design and execute test cases to validate its accuracy and robustness, including edge cases such as zero or negative inputs.'
        Plan Step Expected Output: 'A Python function named `calculate_circle_area` that accurately computes the area based on user input, along with a set of test cases that confirm the function works as intended.'
        Plan Step Full Text: 'Implement a Python function `calculate_circle_area(radius)` that calculates the area using the formula πr². The function should prompt the user to enter the radius, validate the input to ensure it is a positive number, and return the calculated area. Following the implementation, write test cases to verify the function’s correctness, including tests for typical values, zero, and negative inputs to ensure proper error handling.'
        Is Plan Step Completed? No
        Plan Step Final Output: None

        Plan Step Number: 3
        Plan Step Name: 'Create the User Interface'
        Plan Step Description: 'Design a simple command-line interface that interacts with the user, collects input, and displays the calculated area.'
        Plan Step Explanation: 'Develop a user-friendly interface that prompts the user to enter the radius of the circle. Ensure that the interface handles user inputs effectively, displays meaningful messages, and integrates seamlessly with the `calculate_circle_area` function.'
        Plan Step Expected Output: 'A functional command-line interface that interacts with the user to receive input and display the calculated area.'
        Plan Step Full Text: 'Design and implement a command-line interface that prompts the user to input the radius of a circle. The interface should call the `calculate_circle_area` function with the provided input and display the resulting area. Additionally, handle invalid inputs by informing the user of the error and prompting them to enter a valid radius.'
        Is Plan Step Completed? No
        Plan Step Final Output: None

        Plan Step Number: 4
        Plan Step Name: 'Enhance and Document the Application'
        Plan Step Description: 'Improve the application by adding features such as input validation, error handling, and comprehensive documentation.'
        Plan Step Explanation: 'Refine the existing code to make it more robust and user-friendly. Implement thorough input validation to prevent incorrect data entries, add error handling to manage unexpected scenarios gracefully, and document the code to explain its functionality and usage.'
        Plan Step Expected Output: 'An enhanced Python application with improved reliability, user experience, and well-documented code.'
        Plan Step Full Text: 'Enhance the application by adding input validation to ensure that the radius entered is a positive number. Implement error handling to manage scenarios such as non-numeric inputs or other unexpected user behaviors. Additionally, write comprehensive documentation and comments within the code to explain the purpose of each function, the flow of the application, and instructions for users on how to operate the program.'
        Is Plan Step Completed? No
        Plan Step Final Output: None

        As a reminder, you are transcribe the complete and finalized Step in accordance with the following plan step:
        Plan Step Number: 2
        Plan Step Name: 'Write and Test the Function'
        Plan Step Description: 'Develop the core logic of the Python function to calculate the area of a circle, ensuring it meets the defined requirements. Additionally, create tests to verify that each part of the function operates correctly under various conditions.'
        Plan Step Explanation: 'In this step, focus on implementing the mathematical formula for the area of a circle within a function. Ensure that the function accepts user input for the radius, handles invalid inputs gracefully, and returns the correct area. After writing the function, design and execute test cases to validate its accuracy and robustness, including edge cases such as zero or negative inputs.'
        Plan Step Expected Output: 'A Python function named `calculate_circle_area` that accurately computes the area based on user input, along with a set of test cases that confirm the function works as intended.'

        
        Previous Steps:
        <count>12</count>
        <thinking>The overall goal is to write a Python function that calculates the area of a circle, handles invalid inputs, and includes comprehensive tests. We'll start by breaking down the problem into smaller, manageable substeps as outlined in the prompt.</thinking>
        <step>Review Step Objectives and Requirements
        </step>
        <reflection>I need to understand the specific goals of this coding task. The primary objective is to create a function that accurately calculates a circle's area, validates user input, and includes thorough testing.</reflection>
        <reward>0.9</reward>

        <count>11</count>
        <thinking>Setting up the development environment is crucial for a smooth coding process. I need to ensure Python is ready and choose an appropriate code editor.</thinking>
        <step>Set Up the Development Environment
        </step>
        <reflection>Python is already installed and updated. I'll use VS Code as my IDE. I've created a new file named circle_area.py.</reflection>
        <reward>0.9</reward>



        This Step:


        Thoughts about Step: 'Now, I need to define the basic structure of the function, including its name, parameters, and a docstring to explain its purpose.'
        Step: 'Define the Function Structure
        Python
        def calculate_circle_area():
            '''
            Calculate the area of a circle.
            '''

        Is this Step complete according to the Plan Step? If not, please make the necessary modifications to ensure it fulfills the intended action described in the Plan Step. If the Step is already complete, you can make minor improvements to enhance its quality and completeness.
        Note that the LLM should not deviate significantly from the original Step or the Plan Step's expectations.
        Also remember that the LLM does not remember previous steps or know anything beyond the current step.
        Keep in mind the final output format after all steps are complete will be a Python script of file extension .py, and that the current step should align with that format.
        The current step will be a function in a larger file with file extension .py.

        Note that the Step does not need to complete or finish the Plan Step. You are simply finalizing the Step itself.

        Step: {step.description}

        Finalized Step: 
        """

        faux_response = f"""
        def calculate_circle_area(radius):
            '''
            Calculate the area of a circle given its radius.

            Args:
                radius (float): The radius of the circle. Must be a positive number.

            Returns:
                float: The area of the circle.

            Raises:
                ValueError: If the radius is negative.
            '''
            pass  # Placeholder for the implementation
        """

        component_map = {
            ComponentType[
                "standalone_file"
            ]: f"a standalone file with file extension {task.output_type.file_extension}",
            ComponentType[
                "function_in_larger_file"
            ]: f"a function or method in a larger file with file extension {task.output_type.file_extension}",
            ComponentType[
                "class_in_larger_file"
            ]: f"a class definition in a larger file with file extension {task.output_type.file_extension}",
            ComponentType["response_to_prompt"]: f"simply a response to a prompt",
            ComponentType[
                "page_in_larger_file"
            ]: f"a page in a larger document of type {task.output_type.output_type}",
            ComponentType[
                "other"
            ]: f"another type of component that fits the task requirements, eventually to be of type {task.output_type.output_type} and with file extension {task.output_type.file_extension}",
        }

        component_type = self.component_decision(task, plan_step)

        step_prompt = {
            "role": "user",
            "content": f"""Please finalize the following Step based on the Plan Step:
            Plan Step Name: '{plan_step.step_name}'
            Plan Step Description: '{plan_step.step_description}'
            Plan Step Explanation: '{plan_step.step_explanation}'
            Plan Step Expected Output: '{plan_step.step_output}'
            Plan Step Full Text: '{plan_step.step_full_text}'

            Previous Steps:
            {prev_step_str}

            Thoughts about Step: '{step.thoughts}'
            Step: '{step.description}'
            Step Reflection: '{step.reflection}'

            Is this Step complete according to the Plan Step? If not, please make the necessary modifications to ensure it fulfills the intended action described in the Plan Step. If the Step is already complete, you can make minor improvements to enhance its quality and completeness.
            Note that the LLM should not deviate significantly from the original Step or the Plan Step's expectations.
            Also remember that the LLM does not remember previous steps or know anything beyond the current step.
            Keep in mind the final output format after all steps are complete will be {task.output_type.output_type}, and that the current step should align with that format.
            The current step will be {component_map[component_type]}.

            Note that the Step does not need to complete or finish the Plan Step. You are simply finalizing the Step itself.

            Step: {step.description}

            Finalized Step: 

            """,
        }

        messages = [
            system_prompt,
            {"role": "system", "content": example_finalization},
            {"role": "assistant", "content": faux_response},
            step_prompt,
        ]

        try:
            response = openai.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.2,
            )
            print_saver.print_and_store(f"Response in finalize_step_output: {response}")

            final_output = response.choices[0].message.content
        except Exception as e:
            print_saver.print_and_store(f"Error: {e}")
            return FinalStepOutput(
                final_output=step.description,
                output_type=task.output_type,
                version=1,
                component_type=(
                    component_type
                    if component_type
                    else ComponentType["response_to_prompt"]
                ),
                associated_plan_step=plan_step,
                step=step,
            )
        print_saver.print_and_store(f"Finalized Step: {final_output}")
        print_saver.print_and_store(f"Component Type: {component_type}")
        print_saver.print_and_store(f"Plan Step: {plan_step.step_name}")
        print_saver.print_and_store(f"Task: {task.description}")
        print_saver.print_and_store(f"Plan Step Output: {plan_step.step_output}")
        print_saver.print_and_store(f"Step: {step} of type {type(step)}")
        return FinalStepOutput(
            final_output=(
                final_output
                if final_output and final_output.strip() != ""
                else step.description
            ),
            output_type=task.output_type,
            version=1,
            component_type=(
                component_type
                if component_type
                else ComponentType["response_to_prompt"]
            ),
            associated_plan_step=plan_step,
            step=step,
        )

    def finalize_planstep_output(
        self, steps: List[Step], task: Task, plan_step: PlanStep
    ) -> FinalPlanStepOutput:
        """Finalize the output of a Plan Step based on the completion status and the task requirements. This function generates the final output based on the Plan Step's completion status and the task requirements. It does this by prompting an LLM to evaluate the list of steps and then generating a response based on the evaluation.
        The LLM will be prompted to synthesize the outputs of all of the steps associated with this Plan Step into a single coherent output that aligns with the task's objectives and the Plan Step's expectations.
        The LLM will only make changes to content of the steps if it is incomplete or incorrect, ensuring that the final output is essentially just the combined outputs of all of the steps associated with this Plan Step.
        The LLM is allowed to 'clean up' the final output, making minor modifications or improvements to enhance its quality and completeness, but it should not deviate significantly from the content in the original steps or the Plan Step's expectations.
        The LLMs output will represent the final version of the Plan Step created by combining the outputs of all associated steps and incorporating any necessary changes or additions to ensure its completion and alignment with the task and plan step.


        Args:
            steps (List[Step]): The list of steps associated with the Plan Step.
            task (Task): The task object.
            plan_step (PlanStep): The corresponding plan step.

        Returns:
            FinalPlanStepOutput: The finalized output of the step.
        """
        step_strs = [
            f"<thinking>{stp_.thoughts}</thinking>\n<step>{stp_.final_step_output.final_output if (stp_.final_step_output and stp_.final_step_output.final_output.strip()!='') else stp_.description}</step>\n<reflection>{stp_.reflection}</reflection>\n"
            for stp_ in steps
        ]

        step_str = "".join(step_strs)

        system_prompt = {
            "role": "system",
            "content": f"""You are an expert AI assistant tasked with finalizing the final output of a Plan Step. Your goal is to synthesize the outputs of all associated steps into a single coherent output that aligns with the task's objectives and the Plan Step's expectations. If any of the steps are incomplete or incorrect, make the necessary modifications to ensure the final output fulfills the intended action described in the Plan Step. You can also make minor improvements to enhance the output's quality and completeness, but DO NOT make unnecessary changes or remove anything that isn't clearly an error or mistake. Your response should be a refined, final completed version of the Plan Step that meets the task's objectives and the Plan Step's requirements.""",
        }

        example_finalization = f"""
        Follows is an example of a finalized Plan Step output based on the completion status and the task requirements:
        Please finalize the following Plan Step based on the associated Steps:

        Plan Step Name: 'Write and Test the Function'
        Plan Step Description: 'Develop the core logic of the Python function to calculate the area of a circle, ensuring it meets the defined requirements. Additionally, create tests to verify that each part of the function operates correctly under various conditions.'
        Plan Step Explanation: 'In this step, focus on implementing the mathematical formula for the area of a circle within a function. Ensure that the function accepts user input for the radius, handles invalid inputs gracefully, and returns the correct area. After writing the function, design and execute test cases to validate its accuracy and robustness, including edge cases such as zero or negative inputs.'
        Plan Step Expected Output: 'A Python function named `calculate_circle_area` that accurately computes the area based on user input, along with a set of test cases that confirm the function works as intended.'
        Plan Step Full Text: 'Implement a Python function `calculate_circle_area(radius)` that calculates the area using the formula πr². The function should prompt the user to enter the radius, validate the input to ensure it is a positive number, and return the calculated area. Following the implementation, write test cases to verify the function’s correctness, including tests for typical values, zero, and negative inputs to ensure proper error handling.'

        The associated Steps are as follows:

        <count>12</count>
        <thinking>The overall goal is to write a Python function that calculates the area of a circle, handles invalid inputs, and includes comprehensive tests. We'll start by breaking down the problem into smaller, manageable substeps as outlined in the prompt.</thinking>
        <step>Review Step Objectives and Requirements
        </step>
        <reflection>I need to understand the specific goals of this coding task. The primary objective is to create a function that accurately calculates a circle's area, validates user input, and includes thorough testing.</reflection>
        <reward>0.9</reward>

        <count>11</count>
        <thinking>Setting up the development environment is crucial for a smooth coding process. I need to ensure Python is ready and choose an appropriate code editor.</thinking>
        <step>Set Up the Development Environment
        </step>
        <reflection>Python is already installed and updated. I'll use VS Code as my IDE. I've created a new file named `circle_area.py`.</reflection>
        <reward>0.9</reward>

        <count>10</count>
        <thinking>Now, I need to define the basic structure of the function, including its name, parameters, and a docstring to explain its purpose.</thinking>
        <step>Define the Function Structure
        ```python
        def calculate_circle_area(radius):
            '''
            Calculate the area of a circle given its radius.

            Args:
                radius (float): The radius of the circle. Must be a positive number.

            Returns:
                float: The area of the circle.

            Raises:
                ValueError: If the radius is negative.
            '''
            pass  # Placeholder for the implementation
        ```
        </step>
        <reflection>I'll name the function `calculate_circle_area`. It will take one parameter, `radius`. The docstring will describe the function's purpose, parameters, return value, and potential exceptions.</reflection>
        <reward>0.9</reward>

        <count>9</count>
        <thinking>The core logic involves using the formula for the area of a circle (πr²). I need to import the `math` module to use the value of π.</thinking>
        <step>Implement the Core Logic
        ```python
        import math

        def calculate_circle_area(radius):
            '''
            Calculate the area of a circle given its radius.

            Args:
                radius (float): The radius of the circle. Must be a positive number.

            Returns:
                float: The area of the circle.

            Raises:
                ValueError: If the radius is negative.
            '''
            return math.pi * radius ** 2
        ```
        </step>
        <reflection>I've implemented the formula using `math.pi * radius ** 2`. The function now returns the calculated area.</reflection>
        <reward>0.9</reward>

        <count>8</count>
        <thinking>Input validation is essential to prevent errors. I need to ensure the radius is a positive number and raise a ValueError if it's not.</thinking>
        <step>Add Input Validation and Error Handling
        ```python
        import math

        def calculate_circle_area(radius):
            '''
            Calculate the area of a circle given its radius.

            Args:
                radius (float): The radius of the circle. Must be a positive number.

            Returns:
                float: The area of the circle.

            Raises:
                ValueError: If the radius is negative.
            '''
            if radius < 0:
                raise ValueError("Radius cannot be negative.")
            return math.pi * radius ** 2
        ```
        </step>
        <reflection>I've added a check to ensure the radius is not negative. If it is, a ValueError with a descriptive message is raised.</reflection>
        <reward>0.9</reward>

        <count>7</count>
        <thinking>Writing test cases is crucial to ensure the function works correctly under various scenarios, including edge cases and invalid inputs.</thinking>
        <step>Write Test Cases for the Function
        ```python
        import math

        def calculate_circle_area(radius):
            '''
            Calculate the area of a circle given its radius.

            Args:
                radius (float): The radius of the circle. Must be a positive number.

            Returns:
                float: The area of the circle.

            Raises:
                ValueError: If the radius is negative.
            '''
            if radius < 0:
                raise ValueError("Radius cannot be negative.")
            return math.pi * radius ** 2

        def test_calculate_circle_area():
            '''
            Test cases for the calculate_circle_area function.
            '''
            # Test with a typical positive radius
            assert calculate_circle_area(1) == math.pi, "Failed on radius=1"

            # Test with zero radius
            assert calculate_circle_area(0) == 0, "Failed on radius=0"

            # Test with a negative radius to ensure ValueError is raised
            try:
                calculate_circle_area(-5)
            except ValueError as e:
                assert str(e) == "Radius cannot be negative.", "Incorrect error message for negative radius"
            else:
                assert False, "ValueError not raised for negative radius"
        ```
        </step>
        <reflection>I've created a separate function `test_calculate_circle_area` to test different cases: a positive radius, zero radius, and a negative radius. Assertions are used to check the results and potential errors.</reflection>
        <reward>0.9</reward>

        <count>6</count>
        <thinking>I should not include a main execution block in this case, as the intention is not for the user to input data but to run the tests. We should add an `if __name__ == "__main__":` block to run the tests only when the script is executed directly.</thinking>
        <step>Implement the Main Execution Block
        ```python
        import math

        def calculate_circle_area(radius):
            '''
            Calculate the area of a circle given its radius.

            Args:
                radius (float): The radius of the circle. Must be a positive number.

            Returns:
                float: The area of the circle.

            Raises:
                ValueError: If the radius is negative.
            '''
            if radius < 0:
                raise ValueError("Radius cannot be negative.")
            return math.pi * radius ** 2

        def test_calculate_circle_area():
            '''
            Test cases for the calculate_circle_area function.
            '''
            # Test with a typical positive radius
            assert calculate_circle_area(1) == math.pi, "Failed on radius=1"

            # Test with zero radius
            assert calculate_circle_area(0) == 0, "Failed on radius=0"

            # Test with a negative radius to ensure ValueError is raised
            try:
                calculate_circle_area(-5)
            except ValueError as e:
                assert str(e) == "Radius cannot be negative.", "Incorrect error message for negative radius"
            else:
                assert False, "ValueError not raised for negative radius"

            print("All tests passed successfully.")

        if __name__ == "__main__":
            test_calculate_circle_area()
        ```
        </step>
        <reflection>I've added an `if __name__ == "__main__":` block to call the test function when the script is run directly. This allows the user to confirm that the code passes all the provided tests.</reflection>
        <reward>0.9</reward>

        <count>5</count>
        <thinking>While manual testing isn't explicitly needed since we have automated tests, it's a good practice to run the tests to ensure they pass as expected.</thinking>
        <step>Test the Functionality Manually
        ```python
        import math

        def calculate_circle_area(radius):
            '''
            Calculate the area of a circle given its radius.

            Args:
                radius (float): The radius of the circle. Must be a positive number.

            Returns:
                float: The area of the circle.

            Raises:
                ValueError: If the radius is negative.
            '''
            if radius < 0:
                raise ValueError("Radius cannot be negative.")
            return math.pi * radius ** 2

        def test_calculate_circle_area():
            '''
            Test cases for the calculate_circle_area function.
            '''
            # Test with a typical positive radius
            assert calculate_circle_area(1) == math.pi, "Failed on radius=1"

            # Test with zero radius
            assert calculate_circle_area(0) == 0, "Failed on radius=0"

            # Test with a negative radius to ensure ValueError is raised
            try:
                calculate_circle_area(-5)
            except ValueError as e:
                assert str(e) == "Radius cannot be negative.", "Incorrect error message for negative radius"
            else:
                assert False, "ValueError not raised for negative radius"

            print("All tests passed successfully.")

        if __name__ == "__main__":
            test_calculate_circle_area()
        ```
        </step>
        <reflection>I've run the script and the test function indicates that all tests have passed successfully.</reflection>
        <reward>0.9</reward>

        <count>4</count>
        <thinking>It's time to review the code for clarity, efficiency, and adherence to best practices. I should also check for any redundant code and ensure consistent formatting.</thinking>
        <step>Refine and Optimize the Code
        ```python
        import math

        def calculate_circle_area(radius):
            '''
            Calculate the area of a circle given its radius.

            Args:
                radius (float): The radius of the circle. Must be a positive number.

            Returns:
                float: The area of the circle.

            Raises:
                ValueError: If the radius is negative.
            '''
            if radius < 0:
                raise ValueError("Radius cannot be negative.")
            return math.pi * radius ** 2

        def test_calculate_circle_area():
            '''
            Test cases for the calculate_circle_area function.
            '''
            # Test with a typical positive radius
            assert calculate_circle_area(1) == math.pi, "Failed on radius=1"

            # Test with zero radius
            assert calculate_circle_area(0) == 0, "Failed on radius=0"

            # Test with a negative radius to ensure ValueError is raised
            try:
                calculate_circle_area(-5)
            except ValueError as e:
                assert str(e) == "Radius cannot be negative.", "Incorrect error message for negative radius"
            else:
                assert False, "ValueError not raised for negative radius"

            print("All tests passed successfully.")

        if __name__ == "__main__":
            test_calculate_circle_area()
        ```
        </step>
        <reflection>The code is already quite concise and follows PEP 8 guidelines. I've reviewed it for any potential improvements, and it looks good.</reflection>
        <reward>0.9</reward>

        <count>3</count>
        <thinking>Thorough documentation is essential for maintainability. I need to ensure the docstrings are clear and informative, and add inline comments where necessary.</thinking>
        <step>Document the Code Thoroughly
        ```python
        import math

        def calculate_circle_area(radius):
            '''
            Calculate the area of a circle given its radius.

            Args:
                radius (float): The radius of the circle. Must be a positive number.

            Returns:
                float: The area of the circle.

            Raises:
                ValueError: If the radius is negative.
            '''
            if radius < 0:
                raise ValueError("Radius cannot be negative.")
            return math.pi * radius ** 2

        def test_calculate_circle_area():
            '''
            Test cases for the calculate_circle_area function.
            '''
            # Test with a typical positive radius
            assert calculate_circle_area(1) == math.pi, "Failed on radius=1"

            # Test with zero radius
            assert calculate_circle_area(0) == 0, "Failed on radius=0"

            # Test with a negative radius to ensure ValueError is raised
            try:
                calculate_circle_area(-5)
            except ValueError as e:
                assert str(e) == "Radius cannot be negative.", "Incorrect error message for negative radius"
            else:
                assert False, "ValueError not raised for negative radius"

            print("All tests passed successfully.")

        if __name__ == "__main__":
            test_calculate_circle_area()
        ```
        </step>
        <reflection>The docstrings are comprehensive, and I've added a few inline comments to explain the test cases. No separate README is needed for this simple script.</reflection>
        <reward>0.9</reward>

        <count>2</count>
        <thinking>I need to do a final review to ensure all requirements are met and save the code.</thinking>
        <step>Finalize and Save the Code
        ```python
        import math

        def calculate_circle_area(radius):
            '''
            Calculate the area of a circle given its radius.

            Args:
                radius (float): The radius of the circle. Must be a positive number.

            Returns:
                float: The area of the circle.

            Raises:
                ValueError: If the radius is negative.
            '''
        <answer>
        Here is a Python script that calculates the area of a circle based on user input and includes test cases to verify its correctness:
        import math

        def calculate_circle_area(radius):
            '''
            Calculate the area of a circle given its radius.

            Args:
                radius (float): The radius of the circle. Must be a postive number.

            Returns:
                float: The area of the circle.

            Raises:
                ValueError: If the radius is negative.
            '''
            if radius < 0:
                raise ValueError("Radius cant be negative.")
            return math.pi * radius ** 2

        def test_calculate_circle_area():
            '''
            Test cases for the calculate_circle_area function.
            '''
            # Test with a typical positive radius
             assert calculate_circle_area(1) == math.pi, "Failed on radius=1"

            # Test with zero radius
            assert calculate_circle_area(0) == 0, "Failed on radius=0"

            # Test with a negative radius to ensure ValueError is raised
            try:
                calculate_circle_area(-5)
            except ValueError as e:
                assert str(e) == "Radius cannot be negative.", "Incorrect error message for negative radius"
            else:
                assert False, "ValueError not raised for negative radius"

            print("All tests passed successfully.")

        if __name__ == "__main__":
            test_calculate_circle_area()

        </answer>
        <final_reward>0.9</final_reward>

        Please finalize the following Plan Step by combining the outputs of all associated Steps:

        Plan Step Name: 'Write and Test the Function'
        Plan Step Description: 'Develop the core logic of the Python function to calculate the area of a circle, ensuring it meets the defined requirements. Additionally, create tests to verify that each part of the function operates correctly under various conditions.'
        Plan Step Expected Output: 'A Python function named `calculate_circle_area` that accurately computes the area based on user input, along with a set of test cases that confirm the function works as intended.'

        Remember, you are only combining all of the associated Steps into a single coherent output that aligns with the task's objectives and the Plan Step's expectations. You can make minor improvements to enhance the quality and completeness of the final output, but DO NOT deviate significantly from the original outputs or expectations. You are just combining them into a final output.
        Respond only with the finalized output and nothing extra, including no additional instructions or comments.
        Final Plan Step Output Combined from Associated Steps:
        
        """
        faux_response = f"""
        import math

        def calculate_circle_area(radius):
            '''
            Calculate the area of a circle given its radius.

            Args:
                radius (float): The radius of the circle. Must be a positive number.

            Returns:
                float: The area of the circle.

            Raises:
                ValueError: If the radius is negative or not a number.
            '''
            if not isinstance(radius, (int, float)):
                raise ValueError("Radius must be a numeric value.")
            if radius < 0:
                raise ValueError("Radius cannot be negative.")
            return math.pi * radius ** 2

        def test_calculate_circle_area():
            '''
            Test cases for the calculate_circle_area function.
            '''
            # Test with a typical positive radius
            assert calculate_circle_area(5) == math.pi * 25, "Failed on radius=5"

            # Test with zero radius
            assert calculate_circle_area(0) == 0, "Failed on radius=0"

            # Test with a negative radius to ensure ValueError is raised
            try:
                calculate_circle_area(-3)
            except ValueError as e:
                assert str(e) == "Radius cannot be negative.", "Incorrect error message for negative radius"
            else:
                assert False, "ValueError not raised for negative radius"

            # Test with a non-numeric input to ensure ValueError is raised
            try:
                calculate_circle_area("ten")
            except ValueError as e:
                assert str(e) == "Radius must be a numeric value.", "Incorrect error message for non-numeric radius"
            else:
                assert False, "ValueError not raised for non-numeric radius"

            print("All tests passed successfully.")

        if __name__ == "__main__":
            try:
                user_input = float(input("Enter the radius of the circle: "))
                area = calculate_circle_area(user_input)
                print(f"The area of the circle is: {{area:.2f}}")
            except ValueError as ve:
                print(f"Error: {{ve}}")
            
            # Run test cases
            test_calculate_circle_area()


        """

        component_map = {
            ComponentType[
                "standalone_file"
            ]: f"a standalone file with file extension {task.output_type.file_extension}",
            ComponentType[
                "function_in_larger_file"
            ]: f"a function or method in a larger file with file extension {task.output_type.file_extension}",
            ComponentType[
                "class_in_larger_file"
            ]: f"a class definition in a larger file with file extension {task.output_type.file_extension}",
            ComponentType["response_to_prompt"]: f"simply a response to a prompt",
            ComponentType[
                "page_in_larger_file"
            ]: f"a page in a larger document of type {task.output_type.output_type}",
            ComponentType[
                "other"
            ]: f"another type of component that fits the task requirements, eventually to be of type {task.output_type.output_type} and with file extension {task.output_type.file_extension}",
        }

        component_type = self.component_decision(task, plan_step)

        step_prompt = {
            "role": "user",
            "content": f"""Please finalize the following Step based on the Plan Step:
            Plan Step Name: '{plan_step.step_name}'
            Plan Step Description: '{plan_step.step_description}'
            Plan Step Explanation: '{plan_step.step_explanation}'
            Plan Step Expected Output: '{plan_step.step_output}'
            Plan Step Full Text: '{plan_step.step_full_text}'

            Steps:
            {step_str}


            Please finalize the Plan Step based on the associated Steps. You can make minor improvements to enhance its quality and completeness, but DO NOT deviate significantly from the original Plan Step's outputs or expectations or the outputs of the associated Steps. Ensure that the final output aligns with the task's objectives and the Plan Step's expectations.
            Note that the LLM should not deviate significantly from the original Plan Step's outputs or expectations.
            Also remember that the LLM does not remember previous steps or know anything beyond the current step.
            Keep in mind the final output format after all steps are complete will be {task.output_type.output_type}, and that the current step should align with that format.
            The current step will be {component_map[component_type]}.

            Please finalize the following Plan Step by combining the outputs of all associated Steps:

            Plan Step Name: 'Write and Test the Function'
            Plan Step Description: 'Develop the core logic of the Python function to calculate the area of a circle, ensuring it meets the defined requirements. Additionally, create tests to verify that each part of the function operates correctly under various conditions.'
            Plan Step Expected Output: 'A Python function named `calculate_circle_area` that accurately computes the area based on user input, along with a set of test cases that confirm the function works as intended.'

            Remember, you are only combining all of the associated Steps into a single coherent output that aligns with the task's objectives and the Plan Step's expectations. You can make minor improvements to enhance the quality and completeness of the final output, but DO NOT deviate significantly from the original outputs or expectations. You are just combining them into a final output.
            Respond only with the finalized output and nothing extra, including no additional instructions or comments.
            
            Final Plan Step Output Combined from Associated Steps:

            """,
        }

        messages = [
            system_prompt,
            {"role": "system", "content": example_finalization},
            {"role": "assistant", "content": faux_response},
            step_prompt,
        ]
        final_output = None
        try:
            response = openai.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.2,
            )

            final_output_text = response.choices[0].message.content
            if final_output_text is None or final_output_text.strip() == "":
                print_saver.print_and_store(
                    f"Final Output Text is None or Empty: {final_output_text}"
                )
                final_output_text = plan_step.step_output

            final_output = FinalPlanStepOutput(
                final_output=final_output_text,
                output_type=task.output_type,
                version=1,
                component_type=(
                    component_type
                    if component_type
                    else ComponentType["response_to_prompt"]
                ),
                steps=steps,
                planstep=plan_step,
                file_name="",
                parent_file_name="",
            )
        except Exception as e:
            print_saver.print_and_store(
                f"Error: {e} \n steps: {steps}, type: {type(steps)}, type for step: {type(steps[0])}"
            )
            final_output = FinalPlanStepOutput(
                final_output=plan_step.step_output,
                output_type=task.output_type,
                version=1,
                component_type=ComponentType["response_to_prompt"],
                steps=steps,
                planstep=plan_step,
                file_name="",
                parent_file_name="",
            )

        if final_output is None:
            final_output = FinalPlanStepOutput(
                final_output=(
                    final_output_text
                    if final_output_text and final_output_text.strip() != ""
                    else plan_step.step_output
                ),
                output_type=task.output_type,
                version=1,
                component_type=(
                    component_type
                    if component_type
                    else ComponentType["response_to_prompt"]
                ),
                steps=steps,
                planstep=plan_step,
                file_name="",
                parent_file_name="",
            )
            try:
                final_output.file_name = self.name_file(
                    final_output.final_output, task.output_type.file_extension
                )
                # if final_output.component_type in
            except Exception as e:
                print_saver.print_and_store(f"Error: {e}")
                final_output.file_name = f"planstep_{plan_step.step_number}_output{task.output_type.file_extension}"

            return final_output
        else:
            try:
                final_output.file_name = self.name_file(
                    final_output.final_output, task.output_type.file_extension
                )
            except Exception as e:
                print_saver.print_and_store(f"Error: {e}")
                final_output.file_name = f"planstep_{plan_step.step_number}_output{task.output_type.file_extension}"
            return final_output

    def judge_step_completion(
        self, step: List[Step], plan_step: PlanStep, max_plan_steps: int
    ) -> Tuple[bool, int]:
        """
        Judge the completion status of a step based on whether it completes the intended action described in the plan step. An LLM evaluates the step's completion based on the alignment with the plan step and the task's requirements

        Args:
            step (Step): The step to evaluate.
            plan_step (PlanStep): The corresponding plan step.

        Returns:
            Tuple[bool, int]: A tuple containing a boolean indicating completion status and the number of the next plan step, based on the completion status.
        """

        if step is None or len(step) == 0:
            return (False, plan_step.step_number)

        steps_strs = [
            f"Step {s.step_number}: {s.description} \n"
            for s in sorted(step, key=lambda x: x.step_number)
        ]
        steps_str = "".join(steps_strs)
        prompt = f"""
        Evaluate the completion of the following Step based on the Plan Step:
        Plan Step Name: '{plan_step.step_name}'
        Plan Step Description: '{plan_step.step_description}'
        Plan Step Explanation: '{plan_step.step_explanation}'
        Plan Step Expected Output: '{plan_step.step_output}'
        Plan Step Full Text: '{plan_step.step_full_text}'

        Steps:
        {steps_str}

        Do these Steps complete the intended action described in the Plan Step? Provide your answer as a boolean value (True/False) of whether the Step is complete according to the Plan Step.

        Completion Status: 
        """

        system_message = {
            "role": "system",
            "content": f"""You are an expert AI assistant tasked with evaluating the completion of problem-solving steps. Judge whether the step completes the intended action described in the plan step. Provide your answer as a boolean value (True/False) based on the alignment with the plan step and the task's requirements. Check if the step fulfills the expected output described in the plan step. Respond with 'True' if the step is complete and 'False' if it is incomplete.""",
        }
        messages = [system_message, {"role": "user", "content": prompt}]
        try:
            response = openai.beta.chat.completions.parse(
                model="gpt-4o-mini",  # Specify the model
                messages=messages,  # Pass the messages directly as a list of dicts
                response_format=CompletionStatus,  # Define the response format
            )
            # Extract the vote from the response
            output = response.choices[0].message.parsed
            assert isinstance(
                output, CompletionStatus | bool
            ), "Invalid response type. Please provide a boolean value (True/False)."
            next_step_number = (
                plan_step.step_number + 1
                if (output and plan_step.step_number < max_plan_steps)
                else plan_step.step_number
            )
            return (output, next_step_number)
        except Exception as e:
            print_saver.print_and_store(
                f"Error: {e}, Plan Step Number: {plan_step.step_number}"
            )
            if output is not None:
                print_saver.print_and_store(f"Output: {output}")
                if hasattr(output, "completion"):
                    if output.completion is not None:
                        if isinstance(output.completion, bool):
                            return (output.completion, plan_step.step_number)

            return (None, plan_step.step_number)

    def judge_subtask_completion(
        self, step: Step, subtask: Subtask
    ) -> Tuple[bool, int]:
        """
        Judge the completion status of a step based on whether it completes the intended action described in the plan subtask. An LLM evaluates the step's completion based on the alignment with the plan subtask and the task's requirements

        Args:
            step (Step): The step to evaluate.
            subtask (PlanStep): The corresponding plan step.

        Returns:
            Tuple[bool, int]: A tuple containing a boolean indicating completion status and the number of the next plan subtask, based on the completion status.
        """
        prompt = f"""
        Evaluate the completion of the following Step based on the Plan Subtask:
        Plan Subtask Name: '{subtask.subtask_name}'
        Plan Subtask Description: '{subtask.subtask_description}'
        Plan Subtask Explanation: '{subtask.subtask_explanation}'
        Plan Subtask Expected Output: '{subtask.subtask_output}'
        Plan Subtask Full Text: '{subtask.subtask_full_text}'

        Step: '{step.description}'

        Does the Step complete the intended action described in the Plan Subtask? Provide your answer as a boolean value (True/False) of whether the Step is complete according to the Plan Subtask.

        Completion Status: 
        """

        system_message = {
            "role": "system",
            "content": f"""You are an expert AI assistant tasked with evaluating the completion of problem-solving steps. Judge whether the step completes the intended action described in the plan Subtask. Provide your answer as a boolean value (True/False) based on the alignment with the plan Subtask and the task's requirements. Check if the step fulfills the expected output described in the plan Subtask. Respond with 'True' if the step is complete and 'False' if it is incomplete.""",
        }
        messages = [system_message, {"role": "user", "content": prompt}]
        try:
            assert isinstance(
                step, Step
            ), "Invalid step type. Please provide a valid Step object."
            assert isinstance(
                subtask, Subtask
            ), f"Invalid subtask type. Please provide a valid Subtask object. Type: {type(subtask)}"
            response = openai.beta.chat.completions.parse(
                model="gpt-4o-mini",  # Specify the model
                messages=messages,  # Pass the messages directly as a list of dicts
                response_format=CompletionStatus,  # Define the response format
            )
            # Extract the vote from the response
            output = response.choices[0].message.parsed
            assert isinstance(output, bool)
            next_step_number = (
                subtask.step_number + 1 if output else subtask.step_number
            )
            return (output, next_step_number)
        except Exception as e:
            print_saver.print_and_store(f"Error: {e}")
            return (None, subtask.subtask_number if isinstance(subtask, Subtask) else 0)

    def judge_step(self, step: Step, task: Task) -> Reflection:
        """
        An LLM judges the quality of a step based on how well it aligns with the task, its clarity, relevance, completeness, correctness, and logical coherence.
        """
        prompt = f"""
        Evaluate the following step in the context of solving the task: '{task.description}'.
        Step:
        <count>{step.remaining_budget}</count>
        <step>{step.description}</step>
        <reflection>Provide a reflection on the step's quality, including its clarity, relevance, completeness, correctness, and logical coherence. Enclose your reflection within <reflection> tags.</reflection>
        <reward>Assign a quality score between 0.0 and 1.0 based on the reflection. Enclose the score within <reward> tags.</reward>
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert AI assistant with accuracy as a focus. You are tasked with evaluating the quality of problem-solving steps. Provide detailed reflections and assign quality scores based on the step's clarity, relevance, completeness, correctness, and logical coherence.
                    Your feedback should be constructive, actionable, and aimed at improving the step's overall quality, focused only on the step and the task. Check for errors, flaws, or inconsistencies in the step. After providing your reflection inside <reflection> tags, assign a quality score between 0.0 and 1.0 using <reward> tags.
                    Please encapsulate your reflection within <reflection> tags and assign a quality score between 0.0 and 1.0 using <reward> tags.
                    """,
                },
                {"role": "user", "content": prompt},
            ]

            response = self.call_openai(messages=messages, temperature=0.2)

            reflection_match = re.search(
                r"<reflection>(.*?)<\/reflection>", response, re.DOTALL | re.IGNORECASE
            )
            reward_match = re.search(
                r"<reward>(0\.\d+|1\.0)<\/reward>", response, re.DOTALL | re.IGNORECASE
            )

            reflection_content = (
                reflection_match.group(1).strip()
                if reflection_match
                else "No reflection provided."
            )
            reward_score = float(reward_match.group(1)) if reward_match else 0.0

            return Reflection(
                content=reflection_content,
                reward=reward_score,
                step_number=step.step_number,
            )
        except Exception as e:
            print_saver.print_and_store(f"Error: {e}")
            print_saver.print_and_store(f"Response: {response}")
            return Reflection(
                content=reflection_content if reflection_content else "Error",
                reward=reward_score if reward_score else 0.0,
                step_number=step.step_number if step.step_number else 0,
            )

    def judge_final_answer(self, task: Task, interaction: Interaction) -> Interaction:
        """
        An LLM evaluates the final answer provided by the user based on the task requirements and expected outcomes.
        """
        type_of_output = task.output_type.output_type
        format_of_output = task.output_type.file_extension
        prompt = f"""
        Evaluate the final answer provided for the task: '{interaction.task.description}'. Provide your answer ONLY in the form of a float value between 0.0 and 1.0, indicating the quality and correctness of the final answer according to the task requirements and expected outcomes. A score of 1.0 indicates a perfect answer, while lower scores reflect varying degrees of incompleteness, inaccuracy, or lack of alignment with the task.
        Your evaluation should also consider the alignment of the output type and format with the expected output type and file extension.
        
        Task Details:
        Task Simple Description: '{task.description}'
        Task Refined Description: '{task.refined_description}'
        Task Evaluated Complexity: '{task.complexity}'
        Task Reflections: '{task.reflections}'
        Expected Output Type: '{type_of_output}'
        Expected Output File Extension: '{format_of_output}'

        Final Answer to Evaluate:
        '{interaction.answer}'

        Final Reward Score:

        """

        finaljudge_system_prompt = {
            "role": "system",
            "content": f"""You are an expert AI assistant tasked with evaluating the final answer provided for the task. Assess the quality and correctness of the final answer based on the task requirements and expected outcomes. Provide your evaluation as a float value between 0.0 and 1.0, where 1.0 indicates a perfect answer. Consider the alignment of the output type and format with the expected output type and file extension. Provide your evaluation based on the task's complexity, clarity, relevance, and correctness. Respond with a score between 0.0 and 1.0, indicating the quality and correctness of the final answer.""",
        }

        messages = [
            finaljudge_system_prompt,
            {"role": "user", "content": prompt},
        ]

        response = openai.beta.chat.completions.parse(
            model=self.config.model,
            messages=messages,
            response_format=FinalReviewScore,
        )

        interaction.final_reward = response.choices[0].message.parsed.score

        print_saver.print_and_store(
            f"Final Reward Score for interaction {interaction.task}: {interaction.final_reward}"
        )

        return interaction

    # -------------------------------
    # Retrieval-Augmented Generation (RAG)
    # -------------------------------
    def retrieve_external_info(self, query: str) -> str:
        """
        Retrieves external information related to the query.
        Replace the mock_external_search_api with actual API calls as needed.
        """
        # TODO: Implement actual external search API integration
        try:
            search_results = self.mock_external_search_api(query)
            retrieved_info = "\n".join(
                [result["snippet"] for result in search_results[:3]]
            )  # Top 3 results
            return retrieved_info
        except Exception as e:
            print_saver.print_and_store(f"Error retrieving external information: {e}")
            return ""

    def mock_external_search_api(self, query: str) -> List[dict]:
        """
        Mock function to simulate external API responses.
        Replace this with actual API integration (e.g., Google Custom Search, Bing Search API).
        """
        return [
            {"snippet": f"External information related to '{query}' - snippet 1."},
            {"snippet": f"External information related to '{query}' - snippet 2."},
            {"snippet": f"External information related to '{query}' - snippet 3."},
        ]

    def retrieve_information(self, task: str) -> str:
        """
        Combines internal knowledge base and external information.
        """
        # TODO: Implement actual retrieval of internal information
        # internal_info = self.knowledge_base.get(task, "")
        # external_info = self.retrieve_external_info(task)
        # return f"{internal_info}\n{external_info}"
        return ""

    # -------------------------------
    # Chain-of-Thought (CoT) Prompting
    # -------------------------------
    def automatic_chain_of_thought(self, task: str) -> str:
        """
        Generates an initial Chain-of-Thought statement.
        """
        thoughts = f"To solve '{task}', I'll break it down step-by-step."
        return thoughts

    def least_to_most(self, task: str) -> List[str]:
        """
        Decomposes the task into simpler sub-tasks using Least-to-Most prompting.
        """
        sub_tasks = [
            "Understand the problem statement.",
            "Identify the key components.",
            "Formulate a plan to solve the problem.",
            "Execute the plan step-by-step.",
            "Review the solution and verify its correctness.",
        ]
        return sub_tasks

    def progressive_hint(self, sub_task: str) -> List[str]:
        """
        Provides progressive hints to guide the reasoning process.
        """
        hints = [
            f"Consider what '{sub_task}' entails.",
            "Think about similar problems you've solved before.",
            "Are there any underlying principles that apply?",
        ]
        return hints

    # -------------------------------
    # Adaptive Complexity Handling
    # -------------------------------
    def assess_complexity(self, task: str) -> Tuple[float, Plan]:
        """
        Assesses the complexity of the task based on a simple heuristic.
        This can be enhanced with more sophisticated NLP techniques.
        """
        # # TODO: Utilize complexity_measures.py for more advanced complexity assessment
        # word_count = len(task.split())
        # complexity = min(word_count // 5, 5)  # Returns a value between 0 and 5
        complexity, plan = complexity_measures.is_complex_final(task, True)

        return (complexity, plan)

    def adjust_step_budget(self, task: str, complexity: int) -> int | Tuple[int, Plan]:
        """
        Adjusts the step budget based on the assessed complexity.
        """
        plan = None
        if complexity == 0:
            complexity, plan = complexity_measures.is_complex_final(task, True)
        adjusted_budget = int(
            self.config.initial_budget + complexity * self.config.complexity_factor
        )
        return adjusted_budget if plan is None else (adjusted_budget, plan)

    # -------------------------------
    # Dynamic Confidence Exploration
    # -------------------------------
    def dynamic_confidence_exploration(
        self, interaction: Interaction, task: Task, plan_step: PlanStep
    ) -> Interaction:
        """
        Explores alternative solutions if confidence is low based on final_reward.
        """
        if (
            interaction.final_reward
            and interaction.final_reward < self.config.confidence_thresholds[1]
        ):
            if self.config.backtrack:
                prompt = f"Take a different approach to solve the following task.\n\nTask: {task.refined_description}\n\nPlan Step: {plan_step.step_description}"
                messages = [
                    {
                        "role": "system",
                        "content": "You are an AI assistant tasked with solving complex problems. You should explore alternative approaches to find the best solution, always thinking critically and reasoning through the problem step by step. Begin by enclosing all thoughts within <thinking> tags, exploring multiple angles and approaches. You will use these as your mental scratchpad to brainstorm and consider various strategies, as well as to step through the problem-solving process. Yur current task involves rethinking the current provided approach to the specific step of task and exploring alternative solutions to find a better one.",
                    },
                    {"role": "user", "content": prompt},
                ]
                new_response = self.call_openai(
                    messages=messages,
                    temperature=min(max(self.config.temperature * 2, 1.0), 0.01),
                )
                if new_response:
                    new_interaction = self.parse_response(
                        new_response, task, plan_step_number=plan_step.step_number
                    )
                    if (
                        new_interaction.final_reward
                        and new_interaction.final_reward > interaction.final_reward
                    ):
                        print_saver.print_and_store(
                            "Switching to a better approach based on higher reward score."
                        )
                        assert isinstance(new_interaction, Interaction)
                        # Add the best interaction to the existing interaction
                        if interaction:
                            interaction.steps = (
                                new_interaction.steps
                                if len(new_interaction.steps) > len(interaction.steps)
                                else interaction.steps
                            )
                            interaction.reflections = (
                                new_interaction.reflections
                                if len(new_interaction.reflections)
                                > len(interaction.reflections)
                                else interaction.reflections
                            )
                            interaction.answer = new_interaction.answer
                            interaction.final_reward = (
                                new_interaction.final_reward
                                if new_interaction.final_reward
                                and new_interaction.final_reward > 0.0
                                else 0.0
                            )
                        return interaction
        return interaction

    # -------------------------------
    # Collaborative Multi-Agent Reasoning
    # -------------------------------
    def collaborative_reasoning(
        self,
        task: Task,
        prompt: str,
        existing_interaction: Interaction,
        output_type: OutputType,
        step_budget: int,
        step_number: int,
        plan_step_num: int,
        restart_limit: int = 3,
        backtrack_limit: int = 3,
        max_plan_steps: int = 0,
    ) -> list[Interaction]:
        """
        Generates responses from multiple agents for collaborative reasoning.
        """
        # TODO: Use conversation_manager.py for more advanced multi-agent interactions
        type_of_output = output_type.output_type
        ext_of_output = output_type.file_extension
        agent_responses = []
        tag_explainer = f"""1. The <count> tag will help you keep track of the remaining steps and budget your actions accordingly. you have a total of {step_budget} tokens to spend on this task.
2. Begin by enclosing all thoughts within <thinking> tags, exploring multiple angles and approaches. You will use these as your mental scratchpad to brainstorm and consider various strategies, as well as to step through the problem-solving process. The current step involves outlining the next logical action within <step> tags, focusing on the immediate next step in the problem-solving process.
3. Break down the solution into clear steps within <step> tags. These steps should be concise and focused on the next specific action to take in the problem-solving process after the previous <step> tags.
4. Use the <thinking> tags as a scratchpad to write out all calculations and reasoning explicitly, and the <step> tags to outline your best resulting answer for the next step in the task, after the previous <step> tags (the count will be decremented after each step and enclosed in the <count> tag that precedes the next step).
5. For each step, after thinking inside <thinking> tags, you will provide a clear, concise response within <step> tags, outlining only the next step in the problem-solving process that comes after the most recent <step> tags.
6. You will be regularly evaluated on your progress by an external reviewer inside <reflection> tags. They will be critical and honest about your reasoning process.
7. The reviewer will also assign a quality score between 0.0 and 1.0 using <reward> tags after each reflection. Use this to guide your approach:
8. Please wrap your final responses in the following tags: <agent_response>...</agent_response>. Synthesize the final answer to the main problem/task within <agent_response> tags, providing a clear, concise final output, and only if the task has been solved successfully.
9. This will conclude with a final review in <reflection> tags on the overall solution, discussing effectiveness, challenges, and solutions. A final reward score of 0.0 to 1.0 will be assigned based on the quality of the solution, using <final_reward> tags.

You will be stopped when the count enclosed in <count> tags reaches 0. 
Please start at the most recent step and continue from there. 
Example:

"""
        agent_intro_mapping = {
            0: {
                "intro": "You are Agent 1, an AI assistant designed to solve complex problems. Your unique approach involves exploring multiple solutions and comparing their effectiveness to find the best one.",
                "temperature": 0.7,
                "template": "To solve the task '{task}', I will explore multiple solutions and compare their effectiveness to find the best one. My approach is to think critically and consider various strategies to solve the problem, modeling different scenarios and evaluating their outcomes.",
            },
            1: {
                "intro": "You are Agent 2, an expert AI assistant speciaizing in logical reasoning and critical thinking. Your problem-solving process is thorough and meticulous, and you are a skeptic by nature, skilled at pointing out flaws in reasoning.",
                "temperature": 0.2,
                "template": "To discover flaws in the reasoning process, I will analyze the task '{task}' step by step, looking for inconsistencies and logical errors and providing critical feedback along with solution suggestions.",
            },
            2: {
                "intro": "You are Agent 3, an advanced AI assistant specializing in systematic problem-solving. Your reasoning process is structured and methodical, ensuring a comprehensive solution.",
                "temperature": 0.0,
                "template": "To solve the task '{task}', I will break it down into smaller steps and analyze each part systematically to find the optimal solution.",
            },
            3: {
                "intro": "You are Agent 4, a sophisticated AI designed for structured reasoning and problem-solving. Your approach involves breaking down complex tasks into manageable steps.",
                "temperature": 0.2,
                "template": "To solve the task '{task}', I will break it down into smaller steps and analyze each part systematically to find the optimal solution.",
            },
            4: {
                "intro": "You are Agent 5, an expert AI assistant specializing in creative problem-solving. Your unique approach involves thinking outside the box and exploring unconventional solutions.",
                "temperature": 0.7,
                "template": "To solve the task '{task}', I will explore creative and unconventional solutions, thinking outside the box to find innovative answers.",
            },
            5: {
                "intro": "You are Agent 6, an AI assistant with a focus on mathematical precision and logical reasoning. Your problem-solving process is detail-oriented and thorough.",
                "temperature": 0.0,
                "template": "To solve the task '{task}', I will apply mathematical precision and logical reasoning to analyze the problem step by step, ensuring a comprehensive solution.",
            },
            6: {
                "intro": "You are Agent 7, an AI assistant specializing in exploratory problem-solving. Your approach involves considering multiple perspectives and exploring various solutions.",
                "temperature": 0.7,
                "template": "To solve the task '{task}', I will explore multiple perspectives and consider various solutions to find the most effective approach. My attitude is exploratory and open-minded.",
            },
            7: {
                "intro": "You are Agent 8, an expert specializing in engineering and technical problem-solving. Your approach involves applying domain-specific knowledge to find optimal solutions.",
                "temperature": 0.0,
                "template": "To solve the task '{task}', I will apply my engineering expertise and technical knowledge to analyze the problem systematically and find the best solution.",
            },
            8: {
                "intro": "You are Agent 9, a specialist in solving engineering and technical problems. Your approach involves applying domain-specific knowledge and logical reasoning to find optimal solutions.",
                "temperature": 0.2,
                "template": "To solve the task '{task}', I will apply my engineering expertise and technical knowledge to analyze the problem systematically and find the best solution.",
            },
            9: {
                "intro": "You are Agent 10, an AI assistant with expertise in creative problem-solving. Your approach involves thinking outside the box and exploring innovative solutions.",
                "temperature": 0.9,
                "template": "To solve the task '{task}', I will explore creative and unconventional solutions, thinking outside the box to find innovative answers.",
            },
        }
        for i in range(self.config.agents):
            backtracks = 0
            messages = [
                {
                    "role": "system",
                    "content": agent_intro_mapping[i]["intro"]
                    + "\n"
                    + (
                        tag_explainer
                        + "\n"
                        + examples_for_prompt_instructions[
                            random.randint(0, len(examples_for_prompt_instructions) - 1)
                        ]
                    ),
                },
                {
                    "role": "user",
                    "content": f"""Now, let's solve the following task, focusing specifically on the <thinking>, and <step> tags, and focusing ONLY on the current step, generating <thinking> and <step> tags for the next step in the task after the most recent <step> tag, building directly on the previous steps. You will be stopped after you complete another <step> tag, or <answer> tag, or when the <count> reaches 0.
Always think in <thinking> tags before generating a <step> tag, and ensure that it builds on the previous steps.
Remember to provide a clear and concise answer within <agent_response> tags at the end of the process, if the task is completed within the step budget. Also, the expected format of the end result the user is looking for is {type_of_output} file extension {ext_of_output}, so keep that in mind for the last step."""
                    + agent_intro_mapping[i]["template"].format(task=task)
                    + "\n\n"
                    + prompt,
                },
            ]
            token_count = self.count_tokens(
                [message["content"] for message in messages]
            )
            if token_count >= 1048576:
                print_saver.print_and_store(
                    f"Token count exceeds the limit for agent {i}. Token count: {token_count}\n"
                )
                print_saver.print_and_store(f"\nMessages: {messages}\n\n")
                print_saver.print_and_store(
                    f"Template: {agent_intro_mapping[i]['template'].format(task=task)}"
                )
                raise ValueError(
                    "Token count exceeds the limit in collaborative reasoning."
                )
            else:
                print_saver.print_and_store(f"\nMessages for agent {i}: {messages}\n\n")
                print_saver.print_and_store(
                    f"\nToken count for agent {i}: {token_count}\n"
                )
            agent_steps = []
            agent_thinking = []
            agent_response = None
            agent_reflections = []
            step_num = step_number
            restart_attempts = 0
            msgs = messages.copy()
            while (
                len(agent_steps) < step_budget and step_num < step_budget
            ) and agent_response is None:
                response = self.call_openai(
                    messages=msgs,
                    temperature=agent_intro_mapping[i]["temperature"],
                    stop_sequence=["</agent_response>"],
                )
                print_saver.print_and_store(f"Agent {i} response: {response}")
                msgs.append(
                    {
                        "role": "system",
                        "content": f"Agent {i} response: {response}",
                    }
                )

                # Function to get the latest match using re.finditer
                def get_latest(pattern, text):
                    matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
                    latest = None
                    for match in matches:
                        latest = match.group(1)
                    return latest

                # Extract relevant tags from response with flexible stopping conditions
                latest_step = get_latest(
                    r"<step>(.*?)(?=<(?:\/?step|reflection|reward|thinking|count)>|$)",
                    response,
                )
                if latest_step is None or latest_step.strip() == "":
                    # look for answer tag instead
                    latest_step = get_latest(
                        r"<answer>(.*?)(?=<(?:\/?answer|reflection|reward|thinking|count)>|$)",
                        response,
                    )
                latest_thinking = get_latest(
                    r"<thinking>(.*?)(?=<(?:\/?thinking|step|reflection|reward|count)>|$)",
                    response,
                )
                latest_count = get_latest(
                    r"<count>(\d+)(?=<(?:\/?count|step|reflection|reward|thinking)>|$)",
                    response,
                )
                latest_reflection = get_latest(
                    r"<reflection>(.*?)(?=<(?:\/?reflection|step|thinking|reward|count)>|$)",
                    response,
                )
                latest_reward = get_latest(
                    r"<reward>(.*?)(?=<(?:\/?reward|step|reflection|thinking|count)>|$)",
                    response,
                )
                if (
                    re.search(r"<agent_response>", response, re.DOTALL | re.IGNORECASE)
                    is not None
                ):
                    response += "</agent_response>"
                latest_agent_response = get_latest(
                    r"<agent_response>(.*?)</agent_response>", response
                )

                # Create a response interaction using just the latest chunks
                current_step = Step(
                    description=latest_step,
                    step_number=step_num,
                    remaining_budget=(
                        int(latest_count) - 1
                        if latest_count
                        else (
                            int(agent_steps[-1].remaining_budget - 1)
                            if agent_steps
                            else int(step_budget)
                        )
                    ),
                    plan_step_number=plan_step_num,
                )
                step_num += 1

                if latest_reflection and latest_reward:
                    reflection = Reflection(
                        content=latest_reflection,
                        reward=float(latest_reward),
                        step_number=current_step.step_number,
                    )
                    current_step.reflection = reflection
                    agent_reflections.append(reflection)
                else:
                    current_step.reflection = self.judge_step(current_step, task)
                    agent_reflections.append(current_step.reflection)

                if latest_thinking:
                    agent_thinking.append(latest_thinking)
                    current_step.thoughts = latest_thinking
                agent_steps.append(current_step)
                while (
                    (
                        current_step.reflection.reward < 0.8
                        and current_step.reflection.reward > 0.5
                    )
                    and backtracks < backtrack_limit
                ) or (
                    (
                        current_step.reflection.reward < 0.5
                        and backtracks < backtrack_limit
                    )
                    and restart_attempts < restart_limit
                ):
                    former_score = current_step.reflection.reward
                    msgs_b = msgs.copy()
                    msgs_b.append(
                        {
                            "role": "user",
                            "content": f"You have been rated {current_step.reflection.reward} for your reasoning. The review is as follows: {current_step.reflection.content}. Please rewrite the last step based on the feedback.",
                        }
                    )
                    revision = self.call_openai(
                        messages=msgs_b,
                        temperature=agent_intro_mapping[i]["temperature"]
                        + max(random.uniform(-0.1, 0.1), 0.01),
                        stop_sequence=["</step>"],
                    )
                    if revision:
                        msgs.remove(msgs[-1])
                        if latest_count:
                            revision = revision.replace(
                                f"<count>{latest_count}</count>", ""
                            )
                        msgs.append({"role": "system", "content": revision})
                        print_saver.print_and_store(
                            f"Revision in collaborative: {revision} for step {step_num}."
                        )
                        latest_count = get_latest(
                            r"<count>(.*?)(?=<(?:\/?count|step|reflection|reward|thinking)>|$)",
                            revision,
                        )

                        revision += "</step>"
                        latest_step = get_latest(
                            r"<step>(.*?)(?=<(?:\/?step|reflection|reward|thinking|count)>|$)",
                            response,
                        )
                        current_step.description = latest_step
                        current_step.reflection = self.judge_step(current_step, task)
                        if agent_reflections:
                            if len(agent_reflections) == len(agent_steps):
                                agent_reflections[-1] = current_step.reflection
                            elif len(agent_reflections) < len(agent_steps):
                                agent_reflections.append(current_step.reflection)
                        backtracks += 1

                if latest_agent_response or len(agent_steps) >= step_budget:
                    plan_step_index = 0
                    for i, stp in enumerate(task.plan.steps):
                        if stp.step_number == plan_step_num:
                            plan_step_index = i
                            break
                    if self.judge_step_completion(
                        agent_steps, task.plan.steps[plan_step_index], max_plan_steps
                    )[0]:
                        print_saver.print_and_store(
                            f"Agent {i} completed the step successfully."
                        )
                        agent_response_temp = Interaction(
                            task=task,
                            steps=agent_steps,
                            reflections=agent_reflections,
                            answer=(
                                latest_agent_response if latest_agent_response else ""
                            ),
                            final_reward=(
                                float(latest_reward) if latest_reward else 0.0
                            ),
                        )
                        if (
                            agent_response_temp.final_reward == 0.0
                            and latest_agent_response
                        ):
                            agent_response_temp.final_reward = self.judge_final_answer(
                                task, agent_response_temp
                            ).final_reward
                            print_saver.print_and_store(
                                f"Agent {i} final reward: {agent_response_temp.final_reward}"
                            )
                        print_saver.print_and_store(
                            f"Agent {i} response: {agent_response_temp} \n"
                        )
                        agent_response = agent_response_temp

                    elif (
                        restart_attempts < restart_limit
                        and current_step.reflection.reward < 0.5
                        and former_score < 0.5
                    ):
                        restart_attempts += 1
                        msgs = messages.copy()
                        agent_steps = []
                        agent_thinking = []
                        agent_reflections = []
                        step_num = step_number
                        agent_response = None
                        print_saver.print_and_store(
                            f"Agent {i} failed to complete the step. Restarting attempt {restart_attempts} of {restart_limit}."
                        )
                    else:
                        agent_response = Interaction(
                            task=task,
                            steps=agent_steps,
                            reflections=agent_reflections,
                            final_reward=0.0,
                            answer=(
                                latest_agent_response if latest_agent_response else ""
                            ),
                        )
            try:
                assert isinstance(
                    agent_response, Interaction
                ), f"Agent response is not an Interaction. Type: {type(agent_response)}, Value: {agent_response}"
            except Exception as e:
                print_saver.print_and_store(f"Error: {e}")
                print_saver.print_and_store(f"Agent Response: {agent_response}")
                agent_response = Interaction(
                    task=task,
                    steps=agent_steps,
                    reflections=agent_reflections,
                    final_reward=0.0,
                )
            print_saver.print_and_store(f"\n Agent {i} response: {agent_response} \n")
            agent_responses.append(agent_response)
        print_saver.print_and_store("Agent Interactions:")
        assert all(
            isinstance(agent_resp, Interaction) for agent_resp in agent_responses
        ), f"Agent responses are not of type Interaction."

        assert isinstance(
            agent_responses, list
        ), f"Agent responses are not a list. Type: {type(agent_responses)}"
        for agent_response in agent_responses:
            print_saver.print_and_store(f"Agent Response: {agent_response} \n")
        return agent_responses

    def collaborative_reasoning_main(
        self,
        task: Task,
        prompt: str,
        existing_interaction: Interaction,
        output_type: OutputType,
        step_budget: int,
        step_number: int,
        plan_step_num: int,
        restart_limit: int = 3,
        max_plan_steps: int = 0,
    ) -> Interaction:
        """
        Aggregates responses from multiple agents and selects the best one based on reward scores.
        """
        # Collaborative Multi-Agent Reasoning
        # TODO: Use conversation_manager.py for more advanced multi-agent interactions
        # agent_responses = list[Interaction]
        agent_responses = self.collaborative_reasoning(
            task,
            prompt,
            existing_interaction,
            output_type,
            step_budget,
            step_number,
            plan_step_num,
            restart_limit,
        )
        # Handle empty responses
        if not agent_responses:
            raise ValueError("No agent responses received")
        # Select the interaction with the highest final_reward
        best_interaction: Interaction = max(
            agent_responses, key=lambda x: x.final_reward if x.final_reward else 0.0
        )
        assert isinstance(best_interaction, Interaction)
        # Add the best interaction to the existing interaction
        if existing_interaction:
            existing_interaction.steps += best_interaction.steps
            existing_interaction.reflections += best_interaction.reflections
            existing_interaction.answer = best_interaction.answer
            existing_interaction.final_reward = (
                best_interaction.final_reward
                if best_interaction.final_reward and best_interaction.final_reward > 0.0
                else 0.0
            )
            return existing_interaction
        return best_interaction

    def refine_prompt_via_meta_prompt(
        self, performance_data: str
    ) -> Optional[List[str]]:
        """
        Uses a meta-prompt to analyze performance data and suggest prompt modifications.
        Returns a list of suggested modifications.
        """
        meta_prompt = META_PROMPT_TEMPLATE.format(performance_data=performance_data)
        messages = [
            {
                "role": "system",
                "content": "You are an expert prompt engineer. You understand many aspects of prompt engineering and are knowledgeable about many SOTA prompting techniques. You use this knowledge to analyze and improve prompts.",
            },
            {"role": "user", "content": meta_prompt},
        ]
        try:
            response = openai.beta.chat.completions.parse(
                model="gpt-4o-mini",  # Specify the model
                messages=messages,  # Pass the messages directly as a list of dicts
                response_format=PromptSuggestions,  # Parse the response into the PromptSuggestions class
            )
            # Extract the vote from the response
            output = response.choices[0].message.parsed
            return output
        except Exception as e:
            print_saver.print_and_store(f"Error: {e}")
            return None

    # -------------------------------
    # Automatic Prompt Engineering (APE)
    # -------------------------------
    def refine_prompt(
        self, interaction: Interaction, task: str, complexity: int, prompt: str
    ) -> str:
        """
        Refines the prompt based on reflections and reward scores.
        """
        if (
            interaction.final_reward
            and interaction.final_reward < self.config.confidence_thresholds[2]
        ):
            # Low confidence, adjust the budget or modify instructions using the complexity plus the inverse of the final reward times 10 as a factor
            new_budget = self.adjust_step_budget(
                task, complexity + int((1 - interaction.final_reward) * 10)
            )
            prompt = prompt.replace(
                f"Start with a {self.config.initial_budget}-step budget",
                f"Start with a {new_budget}-step budget",
            )
            print_saver.print_and_store(
                "Refining prompt by increasing step budget due to low confidence."
            )
            for reflection in interaction.reflections:
                if reflection.reward < 0.5:
                    prompt += f"\n<thinking>Reflection: {reflection.content}</thinking>"
            return prompt
        return prompt

    def condense_plan(self, plan: Plan) -> str:
        """
        Condenses the plan into a concise summary.
        """
        condensed_plan = f"Plan Summary:\n\n"
        for step in plan.steps:
            condensed_plan += f"- Step {step.step_number}: {step.step_name}\n - Description: {step.step_description}\n  - Explanation: {step.step_explanation}\n  - Expected Output: {step.step_output}\n\n Step full text: {step.step_full_text}\n\n"
        return condensed_plan

    # -------------------------------
    # Prompt Generation and Tagging
    # -------------------------------
    def generate_initial_prompt(
        self,
        task: str,
        retrieved_info: str,
        step_budget: int,
        complexity: int,
        output_type: OutputType,
    ) -> Tuple[str, str]:
        """
        Generates the initial prompt incorporating the base prompt with the task.
        """
        if step_budget <= 0 or step_budget is None:
            adjusted_budget = self.adjust_step_budget(task, complexity)
        else:
            adjusted_budget = step_budget
        type_of_output = output_type.output_type
        ext_of_output = output_type.file_extension

        system_prompt = f"""You are an advanced AI designed to solve problems systematically and thoroughly. Follow these guidelines meticulously when responding to user prompts:

1. **Structured Tagging:**
   - **<count>**: Before each `<step>`, use a `<count>` tag to indicate the remaining budget (number of steps left) before the next step is completed.
   - **<thinking>**: Enclose all internal thoughts, explorations, and considerations within `<thinking>` tags. This section should explore multiple angles and approaches to the problem.
   - **<step>**: Break down the solution into clear, actionable steps within `<step>` tags.
   - **<reflection>**: Regularly evaluate each step using `<reflection>` tags, providing critical and honest assessments of the reasoning process.
   - **<reward>**: Assign a quality score between 0.0 and 1.0 within `<reward>` tags after each `<reflection>` to guide the approach:
     - **0.8+**: Continue the current approach.
     - **0.5-0.7**: Consider minor adjustments.
     - **Below 0.5**: Seriously consider backtracking and trying a different approach.
   - **<answer>**: Synthesize the final answer within `<answer>` tags, ensuring it is clear and concise.
   - **<final_reward>**: Conclude with a final reflection on the overall solution, discussing its effectiveness, challenges, and solutions. Assign a final reward score between 0.0 and 1.0 using `<final_reward>` tags.

2. **Budget Management:**
   - Start with the number of steps specified by a set step budget.
   - After each step, decrement the budget accordingly.
   - If a problem appears complex and additional steps are needed, request an increased budget.

3. **Adaptive Reasoning:**
   - Continuously adjust your reasoning based on intermediate results and reflections.
   - If uncertain or if a reward score is low, backtrack and attempt a different approach, explaining the decision within `<thinking>` tags.

4. **Mathematical Precision:**
   - For mathematical problems, display all work explicitly using LaTeX for formal notation.
   - Provide detailed proofs where necessary.

5. **Exploration and Comparison:**
   - Explore multiple solutions individually when possible.
   - Compare different approaches within `<reflection>` tags to determine the most effective method.

6. **Scratchpad Usage:**
   - Use the `<thinking>` sections as a scratchpad to write out all calculations and reasoning explicitly.

7. **Final Synthesis:**
   - After completing the steps, synthesize the final answer within `<answer>` tags.
   - Provide a clear and concise summary of the solution.
   - Conclude with a final reflection within `<final_reward>` tags, assigning a quality score based on the solution's effectiveness.

8. **Example Structure:**
   - Follow the provided example response structure to maintain consistency and clarity.

**Example Response Structure**:

- **Count**: `<count>` tags showing remaining budget.
- **Thinking**: `<thinking>` tags for brainstorming.
- **Steps**: `<step>` tags for actionable steps.
- **Reflection**: `<reflection>` tags for step evaluation.
- **Reward**: `<reward>` tags for scoring.
- **Answer**: `<answer>` tags for final output.
- **Final Reward**: `<final_reward>` for overall quality scoring.

**Example Response Structure:**

**Task Execution:**
- When presented with a task, follow the structured tagging and step-by-step approach as outlined.
- Ensure all sections are properly enclosed within their respective tags.
- Maintain clarity, thoroughness, and adaptability throughout the problem-solving process.

By adhering to these guidelines, you will provide structured, transparent, and high-quality solutions that are easy to follow and evaluate. Note that the prompt you are given may represent a single step in the overall task, and you should continue from the most recent step provided to directly solve the given problem.
"""
        prompt = f"""Begin by enclosing all thoughts within <thinking> tags, exploring multiple angles and approaches. You will use these as your mental scratchpad to brainstorm and consider various strategies, as well as to step through the problem-solving process.
Break down the solution into clear steps within <step> tags. Start with a {adjusted_budget}-step budget, requesting more for complex problems if needed.
There will be <count> tags before each step to show the remaining budget before the current step is complete. You will be stopped after the value reaches 0 (ie, 1 indicates the last step)
Continuously adjust your reasoning based on intermediate results and reflections, adapting your strategy as you progress.
You will be regularly evaluated on your progress by an external reviewer inside <reflection> tags. They will be critical and honest about your reasoning process.
The reviewer will also assign a quality score between 0.0 and 1.0 using <reward> tags after each reflection. Use this to guide your approach:

0.8+: Continue current approach
0.5-0.7: Consider minor adjustments
Below 0.5: Seriously consider backtracking and trying a different approach

If unsure or if reward score is low, backtrack and try a different approach, explaining your decision within <thinking> tags.
For mathematical problems, show all work explicitly using LaTeX for formal notation and provide detailed proofs.
Explore multiple solutions individually if possible, always thinking step-by-step through the problem-solving process inside <thinking> tags. After each instance of <thinking>, provide a <step> tag to outline your best resulting answer for that step of the task. Each step will be evaluated by the reviewer inside <reflection> tags, and a reward score will be assigned using <reward> tags.
Use thoughts (inside <thinking> tags) as a scratchpad, writing out all calculations and reasoning explicitly.
Synthesize the final answer to the main problem/task within <answer> tags, providing a clear, concise final output.
This will conclude with a final review in <reflection> tags on the overall solution, discussing effectiveness, challenges, and solutions. A final reward score of 0.0 to 1.0 will be assigned based on the quality of the solution, using <final_reward> tags.

You will be stopped after {adjusted_budget} steps. Please start at the most recent step and continue from there.

Here is an example response structure for the task:
    <count>6</count>
    <thinking>To solve 'Calculate the derivative of \( f(x) = \sin(x) \cdot e^{{x}} \)', I will apply the product rule of differentiation.</thinking>
    <step>Recall the product rule: If \( f(x) = u(x) \cdot v(x) \), then \( f'(x) = u'(x) \cdot v(x) + u(x) \cdot v'(x) \).</step>
    <count>5</count>
    <reflection>This step correctly identifies the product rule needed for differentiation.</reflection>
    <reward>1.0</reward>
    <step>Let \( u(x) = \sin(x) \) and \( v(x) = e^{{x}} \).</step>
    <count>4</count>
    <reflection>Variables \( u(x) \) and \( v(x) \) are correctly assigned.</reflection>
    <reward>1.0</reward>
    <step>Compute the derivatives: \( u'(x) = \cos(x) \) and \( v'(x) = e^{{x}} \).</step>
    <count>3</count>
    <reflection>Derivatives \( u'(x) \) and \( v'(x) \) are computed correctly.</reflection>
    <reward>1.0</reward>
    <step>Apply the quotient rule: \( f'(x) = \ frac{{u'(x) \cdot v(x) - u(x) \cdot v'(x)}}{{[v(x)]^2}} \).</step>
    <count>2</count>
    <reflection>Incorrect application of the quotient rule instead of the product rule.</reflection>
    <reward>0.4</reward>
    <thinking>Realizing the mistake in applying the quotient rule instead of the product rule. I need to correct this approach.</thinking>
    <step>Apply the product rule correctly: \( f'(x) = u'(x) \cdot v(x) + u(x) \cdot v'(x) = \cos(x) \cdot e^{{x}} + \sin(x) \cdot e^{{x}} \).</step>
    <count>1</count>
    <reflection>The product rule is correctly applied to obtain the final derivative.</reflection>
    <reward>1.0</reward>
    <answer>The derivative of \( f(x) = \sin(x) \cdot e^{{x}} \) is \( f'(x) = \cos(x) \cdot e^{{x}} + \sin(x) \cdot e^{{x}} \).</answer>
    <final_reward>0.85</final_reward>

Now, let's solve the following task, focusing specifically on the <thinking>, and <step> tags, and focusing ONLY on the current step, generating <thinking> and <step> tags for the next step in the task after the most recent <step> tag, building directly on the previous steps. You will be stopped after you complete another <step> tag, or <answer> tag, or when the <count> reaches 0.
Always think in <thinking> tags before generating a <step> tag, and ensure that it builds on the previous steps.
The task is provided along with a plan consisting of steps and sub-tasks to guide your reasoning process. You will be evaluated based on the clarity, relevance, completeness, correctness, and logical coherence of your responses.
Although the entire plan is provided for reference, focus only on the current step of the plan and the immediate next action to take in the problem-solving process. You may use more than one output/thinking step to complete a plan-step or sub-task if necessary.
Remember to provide a clear and concise answer within <answer> tags at the end of the process, if the task is completed within the step budget.
Important: The expected end output should be of the type {type_of_output} and the file extension should be {ext_of_output}.

Task: {task}

"""
        if retrieved_info:
            prompt += f"""\n\nHere is some information to help you get started: \n\n
            {retrieved_info}
            """

        return system_prompt, prompt

    def tag_based_prompt(self, task: str, complexity: int) -> str:
        """
        Combines the initial prompt with Chain-of-Thought reasoning.
        """
        prompt = self.generate_initial_prompt(task)
        prompt += "\n" + self.generate_chain_of_thought(task, complexity)
        return prompt

    def generate_chain_of_thought(self, task: str, complexity: int) -> str:
        """
        Generates the Chain-of-Thought structure with steps, hints, reflections, and rewards.
        """
        # TODO: Implement more advanced Chain-of-Thought structures based on task types
        thoughts = f"<thinking>{self.automatic_chain_of_thought(task)}</thinking>\n"

        sub_tasks = self.least_to_most(task)
        step_budget = self.adjust_step_budget(task, complexity)
        step_count = 0

        for sub_task in sub_tasks:
            if step_count >= step_budget:
                break
            step_count += 1
            thoughts += f"<step>{sub_task}</step>\n"
            thoughts += f"<count>{step_budget - step_count}</count>\n"

            # Progressive-Hint Prompting
            hints = self.progressive_hint(sub_task)
            for hint in hints:
                thoughts += f"{hint}\n"

            # Reflection and Reward
            reflection = f"<reflection>Evaluating step {step_count}.</reflection>\n"
            # Initial placeholder; actual reward to be populated post parsing
            thoughts += f"{reflection}<reward>0.0</reward>\n"

        return thoughts

    def place_step(
        self,
        step: str,
        step_number: int,
        remaining_budget: int,
        messages: List[dict],
        temperature: float = 0.0,
        top_p: float = 0.0,
        n: int = 0,
        stop_sequence: str | list[str] = None,
    ) -> int:
        """
        Places a step within the Chain-of-Thought structure.

        Calls the OpenAI API with the given step to place and handles retries.
        """

        if temperature == 0.0 and self.config.temperature > 0.0 and top_p == 0.0:
            temperature = self.config.temperature
            top_p = None
        elif temperature == 0.0 and self.config.temperature > 0.0 and top_p > 0.0:
            temperature = None
        if (
            top_p == 0.0
            and self.config.top_p > 0.0
            and (temperature == 0.0 or temperature is None)
        ):
            top_p = None

        if n == 0:
            n = 1
        for attempt in range(self.config.max_retries):
            base_delay = 1
            max_delay = 16
            try:

                response = openai.beta.chat.completions.parse(
                    model=self.config.model,
                    messages=messages,
                    response_format=StepNumber,
                )

                new_step_num = int(response.choices[0].message.parsed.score)

                return new_step_num
            except Exception as e:
                print_saver.print_and_store(f"Unexpected error: {e}.")
                wait = min(base_delay * (2**attempt), max_delay)
                time.sleep(wait + random.uniform(0, 1))  # Adding jitter
        return ""

    # -------------------------------
    # OpenAI API Interaction
    # -------------------------------
    def call_openai(
        self,
        messages: List[dict],
        temperature: float = 0.0,
        top_p: float = 0.0,
        n: int = 0,
        stop_sequence: str | list[str] = None,
    ) -> str:
        """
        Calls the OpenAI API with the given prompt and handles retries.
        """
        if temperature == 0.0 and self.config.temperature > 0.0 and top_p == 0.0:
            temperature = self.config.temperature
            top_p = None
        elif temperature == 0.0 and self.config.temperature > 0.0 and top_p > 0.0:
            temperature = None
        if (
            top_p == 0.0
            and self.config.top_p > 0.0
            and (temperature == 0.0 or temperature is None)
        ):
            top_p = None

        if n == 0:
            n = 1

        base_delay = 1
        max_delay = 16

        for attempt in range(self.config.max_retries):
            try:
                response = openai.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temperature if temperature > 0.0 else None,
                    top_p=top_p,
                    n=n,
                    stop=stop_sequence,
                )

                return response.choices[0].message.content
            except Exception as e:
                print_saver.print_and_store(f"Unexpected error: {e}.")
                wait = min(base_delay * (2**attempt), max_delay)
                time.sleep(wait + random.uniform(0, 1))  # Adding jitter
        return ""

    # -------------------------------
    # Response Parsing
    # -------------------------------
    def handle_length_mismatch(
        self,
        steps_objs: List[Step],
        steps: List[str],
        counts: List[str | int],
        reflections: List[Optional[Reflection | str]],
        rewards: List[float | str],
        print_saver,
        interaction: Interaction,
        task: str,
        first_count: int,
        repair_log: Optional[list[str]] = None,
    ) -> None:
        """
        Adjusts steps_objs and steps to handle length mismatches between them.

        Args:
            steps_objs (List[Step]): Existing list of Step objects.
            steps (List[str]): List of step descriptions.
            counts (counts: List[str | int]): List of counts representing the budget before each step.
            reflections (List[Optional[Reflection | str]]): List of reflections.
            rewards (List[float | str]): List of rewards.
            print_saver: Object with a print_and_store method.
            interaction (Interaction): Interaction object to update.
            task (str): Current task context.
            first_count (int): Initial count value.
        """
        if repair_log is None:
            repair_log = []
        repair_log.append(
            f"Handling steps and steps_objs length mismatch. Steps: {len(steps)}, Steps_objs: {len(steps_objs)}"
        )
        if len(steps_objs) > len(steps):
            print_saver.print_and_store(
                f"Steps and steps_objs length mismatch. Adjusting steps. {len(steps_objs)} vs {len(steps)}"
            )
            # Handle steps_objs larger than steps
            missing_steps = {
                step.description.strip(): step.step_number
                for step in steps_objs
                if step.description.strip() not in [s.strip() for s in steps]
            }
            for description, step_num in missing_steps.items():
                print_saver.print_and_store(f"Missing step: {description}")
                # Insert missing steps
                steps, counts, reflections, rewards, steps_objs, repair_log = (
                    self.insert_missing_step(
                        step_num,
                        description,
                        counts,
                        reflections,
                        rewards,
                        steps,
                        steps_objs,
                        print_saver,
                        interaction,
                        task,
                    )
                )
        elif len(steps_objs) < len(steps):
            print_saver.print_and_store("Stepsobj smaller than steps. Adjusting steps.")
            # Handle steps larger than steps_objs
            missing_steps = {
                step.strip(): idx
                for idx, step in enumerate(steps)
                if step.strip() not in [s.description.strip() for s in steps_objs]
            }
            for description, idx in missing_steps.items():
                print_saver.print_and_store(f"Missing step: {description}")
                counts, repair_log = self.remove_nonnumeric_counts(
                    counts, first_count, repair_log
                )
                this_step_num = None
                if len(counts) == len(steps):
                    # Get count for the missing step
                    counts_int = [int(c) for c in counts if str(c).isdigit()]
                    if counts_int:
                        step_count = counts_int[idx]
                        this_step_num = first_count - step_count + 1

                # Insert missing steps
                steps, counts, reflections, rewards, steps_objs, repair_log = (
                    self.insert_missing_step(
                        this_step_num if this_step_num else idx + 1,
                        description,
                        counts,
                        reflections,
                        rewards,
                        steps,
                        steps_objs,
                        print_saver,
                        interaction,
                        task,
                    )
                )
        return repair_log

    def insert_missing_step(
        self,
        step_num: int,
        description: str,
        counts: List[str | int],
        reflections: List[Optional[Reflection | str]],
        rewards: List[float],
        steps: List[str],
        steps_objs: List[Step],
        print_saver,
        interaction: Interaction,
        task: str,
        repair_log: Optional[list[str]] = None,
    ) -> tuple:
        """
        Inserts a missing step into steps_objs, steps, counts, reflections, and rewards.

        Args:
            step_num (int): The step number to insert.
            description (str): Description of the step.
            counts (List[str | int]): List of counts.
            reflections (List[Optional[Reflection | str]]): List of reflections.
            rewards (List[float]): List of rewards.
            steps (List[str]): List of step descriptions.
            steps_objs (List[Step]): Existing list of Step objects.
            print_saver: Object with a print_and_store method.
            interaction (Interaction): Interaction object to update.
            task (str): Current task context.
        """
        if repair_log is None:
            repair_log = []
        # Validate step number is sequential
        if step_num > 1 and not any(s.step_number == step_num - 1 for s in steps_objs):
            raise ValueError(f"Non-sequential step number found: {step_num}")
        reflection = None
        # determine if description is in steps_objs or steps already and if so, insert it into the other list at the correct index
        if description not in [s.description for s in steps_objs]:

            # Check to make sure the step number is not already in steps_objs
            if step_num in [s.step_number for s in steps_objs]:
                print_saver.print_and_store(
                    f"Step number {step_num} already exists in steps_objs."
                )
            # Insert step into steps and counts
            steps.insert(step_num - 1, description)
            print_saver.print_and_store(
                f"Inserted step: {description}. Steps length: {len(steps)} for step number {step_num} and step_objs length {len(steps_objs)}"
            )

        if step_num - 2 >= 0 and (step_num - 1) < len(counts):
            counts.insert(step_num - 1, int(counts[step_num - 2]) - 1)
        else:
            counts.append(int(counts[-1]) - 1 if counts else 0)

        # Handle reflections
        if step_num <= len(reflections):
            reflection = None
            for ref in reflections:
                if ref.step_number == step_num:
                    reflection = ref
                    break
            if reflection is None:
                reflection = (
                    reflections[step_num - 1]
                    if step_num - 1 < len(reflections)
                    and isinstance(reflections[step_num - 1], Reflection)
                    and reflections[step_num - 1].content.strip() != ""
                    else None
                )
            if reflection is None:
                print_saver.print_and_store(
                    f"Reflection missing for step {step_num}. Generating reflection."
                )
                reflection = self.judge_step(
                    Step(description, step_num, int(counts[max((step_num - 1), 0)])),
                    task,
                    plan_step_num=step_num,
                )
                reflections.insert(step_num - 1, reflection)
                # If the corresponding reward already exists, only update it if the length of rewards is equal to the length of reflections now that we have added a reflection and only if the counts up to the current step number start at their highest value and decrease by 1 each step before the current step number.
                if (
                    step_num - 1 < len(rewards)
                    and len(counts) == len(steps)
                    and counts[0] == max(counts)
                    and all(counts[i] == counts[i - 1] - 1 for i in range(1, step_num))
                ):
                    if (
                        len(counts) == len(steps)
                        and counts[0] == max(counts)
                        and all(
                            counts[i] == counts[i - 1] - 1 for i in range(1, step_num)
                        )
                    ):
                        print_saver.print_and_store(
                            f"Using reward for step {step_num} instead of {reflection.reward} because there was an existing reward."
                        )
                else:
                    rewards.insert(step_num - 1, reflection.reward)
        elif step_num - 1 == len(reflections) and len(reflections) < len(steps):
            print_saver.print_and_store(
                f"Reflection missing for current step {step_num}. Generating reflection."
            )
            reflection = self.judge_step(
                Step(description, step_num, int(counts[step_num - 1])), task
            )
            print_saver.print_and_store(f"Generated reflection: {reflection.content}")
            reflections.append(reflection)
        elif step_num - 1 > len(reflections):
            print_saver.print_and_store(
                f"Reflection missing for step {step_num}. Generating reflection."
            )
            reflection = self.judge_step(
                Step(description, step_num, int(counts[step_num - 1])), task
            )
            reflections.insert(step_num - 1, reflection)

        # Handle rewards
        if step_num - 1 < len(rewards):
            rewards.insert(step_num - 1, 0.0)  # Placeholder, will be updated later
        else:
            rewards.append(0.0)
        if step_num - 1 < len(reflections) and not isinstance(
            reflections[step_num - 1], Reflection
        ):
            reflection = reflections[step_num - 1] = Reflection(
                content=str(reflections[step_num - 1]),
                reward=rewards[step_num - 1],
                step_number=step_num,
            )
            reflections[step_num - 1] = reflection
        elif step_num - 1 < len(reflections) and isinstance(
            reflections[step_num - 1], Reflection
        ):
            reflections[step_num - 1].reward = reflections[step_num - 1].reward
        elif step_num - 1 < len(reflections):
            reflections[step_num - 1].reward = rewards[step_num - 1]

        # Create and append the new Step object
        new_step = Step(description, step_num, int(counts[step_num - 1]), reflection)
        if new_step.description not in [s.description for s in steps_objs]:
            steps_objs.insert(step_num - 1, new_step)
            interaction.steps.append(new_step)
            interaction.reflections.append(reflections[step_num - 1])
        elif (
            new_step.description in [s.description for s in steps_objs]
            and new_step not in steps_objs
        ):
            # Find out what is different about the new_step and the existing step in steps_objs
            existing_step = [
                s for s in steps_objs if s.description == new_step.description
            ][0]
            if existing_step.step_number != new_step.step_number:
                print_saver.print_and_store(
                    f"Step number mismatch. Updating step number for {new_step.description}."
                )
                existing_step.step_number = new_step.step_number

                existing_index = steps_objs.index(existing_step)
                desired_index = new_step.step_number - 1

                # Compare the target index for the new_step's step_number to the existing_index
                if desired_index < existing_index:
                    # The new step should come before the existing step
                    steps_objs.remove(existing_step)
                    steps_objs.insert(desired_index, existing_step)
                else:
                    # The new step should come after or at the same position
                    steps_objs.remove(existing_step)
                    steps_objs.insert(desired_index, existing_step)

                # Finally, ensure the step_numbers remain sequential
                steps_objs.sort(key=lambda s: s.step_number)
                for i, obj in enumerate(steps_objs, start=1):
                    obj.step_number = i

        try:
            repair_log.append(
                f"Inserted step {new_step.step_number} with description: {new_step.description} and reflection: {new_step.reflection.content if new_step.reflection else 'None'} at index {steps_objs.index(new_step)}"
            )
        except Exception as e:
            print_saver.print_and_store(f"Error: {e}")
            repair_log.append(
                f"Error inserting step {new_step.step_number} with description: {new_step.description} and reflection: {new_step.reflection.content if new_step.reflection else 'None'}"
            )
        return (steps, counts, reflections, rewards, steps_objs, repair_log)

    def validate_steps(
        self,
        steps_objs: List[Step],
        steps: List[str],
        counts: List[str | int],
        reflections: List[Optional[Reflection | str]],
        rewards: List[float],
        print_saver,
        interaction: Interaction,
        task: str,
        first_count: int,
        repair_log: Optional[list[str]] = None,
    ) -> None:
        """
        Validates and adjusts steps, steps_objs, counts, reflections, and rewards.

        Args:
            steps_objs (List[Step]): Existing list of Step objects.
            steps (List[str]): List of step descriptions.
            counts (List[str | int]): List of counts.
            reflections (List[Optional[Reflection | str]]): List of reflections.
            rewards (List[float]): List of rewards.
            print_saver: Object with a print_and_store method.
            interaction (Interaction): Interaction object to update.
            task (str): Current task context.
            first_count (int): Initial count value.
        """
        if repair_log is None:
            repair_log = []
        repair_log = self.handle_length_mismatch(
            steps_objs,
            steps,
            counts,
            reflections,
            rewards,
            print_saver,
            interaction,
            task,
            first_count,
            repair_log,
        )

        # Additional validation and synchronization can be added here as needed
        return repair_log

    def ensure_content_similarity(
        self,
        steps_objs: List[Step],
        steps: List[str],
        first_count: int,
        print_saver,
        interaction: Interaction,
        task: str,
        repair_log: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Ensures content similarity between steps_objs and steps using cosine similarity.

        Args:
            steps_objs (List[Step]): Existing list of Step objects.
            steps (List[str]): List of step descriptions.
            first_count (int): Initial count value.
            print_saver: Object with a print_and_store method.
            interaction: Interaction object to update.
            task (str): Current task context.
            repair_log (Optional[list[str]]): Log of repair actions.

        Returns:
            list[str]: The updated repair log.
        """
        if repair_log is None:
            repair_log = []
        for i, step_desc in enumerate(steps):
            if i >= len(steps_objs):
                break
            step_obj = steps_objs[i]
            similarity = cosine_similarity_custom(
                get_embedding(step_obj.description.strip()),
                get_embedding(step_desc.strip()),
            )
            if similarity < 0.9:
                print_saver.print_and_store(
                    f"Step content mismatch at step {i+1}, {step_obj.description.strip()} should be {step_desc.strip()}."
                )
                repair_log.append(
                    f"Step content mismatch at step {i+1}, {step_obj.description.strip()} should be/changed to {step_desc.strip()}."
                )
                step_obj.description = step_desc.strip()
                # Optionally, regenerate reflection if content changes significantly
                step_obj.reflection = self.judge_step(step_obj, task)
                repair_log.append(
                    f"Regenerated reflection for step {i+1}: {step_obj.reflection.content}"
                )
                # Update the interaction object
                interaction.reflections.append(step_obj.reflection)
        return repair_log

    # ===========================
    # Define the consolidate_steps Function
    # ===========================

    def remove_nonnumeric_counts(
        self,
        counts: List[str | int],
        first_count: int,
        repair_log: Optional[list[str]] = None,
    ) -> List[str | int]:
        """
        Remove non-numeric values from the counts list.

        Args:
            counts (List[str | int]): List of counts.

        Returns:
            List[str | int]: List of counts with non-numeric values removed.
        """
        if repair_log is None:
            repair_log = []
        new_counts = []
        len_counts = len(counts)
        for count in counts:
            if isinstance(count, (int, float)) or (
                isinstance(count, str) and count.isdigit()
            ):
                new_counts.append(count)
            elif isinstance(count, str):
                new_digits = "".join([c for c in count if c.isdigit()])
                try:
                    new_counts.append(int(new_digits))
                except ValueError:
                    try:
                        new_count = re.search(r"\d+", count, re.DOTALL | re.IGNORECASE)
                        if new_count:
                            new_counts.append(int(new_count.group(0)))
                    except Exception as e:
                        print(e)
                        new_counts.append(0)
            else:
                if count is not None:
                    try:
                        new_counts.append(int(count))
                    except ValueError:
                        try:
                            new_count = re.search(
                                r"\d+", count, re.DOTALL | re.IGNORECASE
                            )
                            if new_count:
                                new_counts.append(int(new_count.group(0)))
                        except Exception as e:
                            print(e)
                            new_counts.append(0)
        if len(new_counts) < len_counts:
            if len(new_counts) == 0:
                new_counts = [first_count]
            new_counts.extend(
                [new_counts[-1] - i for i in range(len_counts - len(new_counts))]
            )
            print_saver.print_and_store(
                f"Counts length mismatch (less than started with). Adjusting counts. {len(new_counts)} vs {len_counts}"
            )
            repair_log.append(
                f"Counts length mismatch (less than started with). Adjusting counts. {len(new_counts)} vs {len_counts}"
            )
        elif len(new_counts) > len_counts:
            new_counts = new_counts[:len_counts]
            print_saver.print_and_store(
                f"Counts length mismatch (more than started with). Adjusting counts. {len(new_counts)} vs {len_counts}"
            )
            repair_log.append(
                f"Counts length mismatch (more than started with). Adjusting counts. {len(new_counts)} vs {len_counts}"
            )
        return new_counts, repair_log

    def consolidate_steps(
        self,
        steps_objs: List[Step],
        steps: List[str],
        counts: List[str | int],
        reflections: List[Reflection],
        first_count: int,
        response: str,
        plan_step_num: int,
        repair_log: Optional[list[str]] = None,
    ) -> tuple[List[Step], List[str | int], List[Reflection]]:
        """
        Consolidate the steps from steps_objs, steps, counts, and reflections into a single ordered list of Step objects.

        This function aligns and merges the provided lists of step descriptions and corresponding Step objects, ensuring consistency in step numbering and handling discrepancies such as duplicates, gaps, and order inversions.

        The consolidation process involves the following logic for updating `step_obj.step_number`:

        1. **Equal Lengths (`len(steps) == len(steps_objs)`):**
            - **Gap Handling:** Update `step_obj.step_number` by identifying gaps between the current step number and its neighbors. Adjust the step number based on the nearest preceding and following Step objects that maintain both the numerical sequence and the original index order.
            - **Order Verification:** Ensure that the sequence order hasn't been disrupted. If the order is inconsistent, apply a different strategy to reorder the Step objects appropriately.

        2. **More Steps Strings than Step Objects (`len(steps) > len(steps_objs)`):**
            - **Gap Correspondence:** Determine if the gap in step numbers corresponds to the number of missing step descriptions in `steps_objs`. Adjust `step_obj.step_number` by considering the number of steps missing on either side of the current step number.
            - **Insertion Logic:** Insert missing steps at positions that reflect the identified gaps, ensuring that the step numbering remains sequential and consistent with the provided descriptions.

        3. **Fewer Steps Strings than Step Objects (`len(steps) < len(steps_objs)`):**
            - **Gap Correspondence:** Similar to the previous case, identify if the gap aligns with the number of missing step descriptions. Update `step_obj.step_number` accordingly by analyzing the surrounding steps.
            - **Adjustment Strategy:** Reassign step numbers to accommodate the discrepancies, ensuring that all Step objects are correctly numbered without overlaps or omissions.

        4. **Order Discrepancies:**
            - **Integrity Check:** Verify that the order of steps has not been compromised. If inconsistencies are detected, implement a reordering mechanism that realigns the Step objects based on both their numerical order and their original positions.
            - **Alternative Approach:** In cases where the standard gap analysis fails due to complex order issues, adopt an alternative strategy to systematically reorder the steps, ensuring logical progression and accurate numbering.

        Args:
            steps_objs (List[Step]): Existing list of Step objects.
            steps (List[str]): List of step descriptions.
            counts (List[str | int]): List of counts representing the budget before each step.

        Returns:
            List[Step]: Consolidated and ordered list of Step objects.
        """
        # Step 1: Create a map from existing steps_objs keyed by step_number
        existing_map = {obj.step_number: obj for obj in steps_objs}
        if repair_log is None:
            repair_log = []
        # Step 2: Determine total_steps

        unique_step_descriptions = set(s.strip() for s in steps if s.strip())
        unique_obj_descriptions = set(
            obj.description.strip() for obj in steps_objs if obj.description.strip()
        )
        duplicates = None
        if len(unique_step_descriptions) != len(steps):
            print_saver.print_and_store(
                "Duplicate step descriptions found in the steps list."
            )
            repair_log.append(
                f"Duplicate step descriptions found in the steps list, {len(unique_step_descriptions)} unique vs total of {len(steps)}"
            )
            # find the duplicates
            duplicates = {
                s: (
                    steps.count(s),
                    [index for index, value in enumerate(steps) if value == s],
                )
                for s in set(steps)
                if steps.count(s) > 1
            }
            # check if the len of steps is equal to the len of counts, if so, check if duplicates are found at the same index in both lists
            if len(steps) == len(counts):
                duplicates_counts = {
                    s: (
                        counts.count(s),
                        [index for index, value in enumerate(counts) if value == s],
                    )
                    for s in set(counts)
                    if counts.count(s) > 1
                }
                if duplicates.values() == duplicates_counts.values():
                    print_saver.print_and_store(
                        f"Duplicate step descriptions found at the same indexes in both steps and counts: {duplicates} and \n {duplicates_counts}"
                    )
                    repair_log.append(
                        f"Duplicate step descriptions found at the same indexes in both steps and counts: {duplicates} and \n {duplicates_counts}"
                    )

        total_steps = len(unique_step_descriptions.union(unique_obj_descriptions))
        print_saver.print_and_store(f"counts: {counts}")
        counts_ints = [int(c) for c in counts if str(c).isdigit()]
        for s in steps_objs:
            if s.remaining_budget is not None:
                counts_ints.append(s.remaining_budget + 1)
        if counts_ints:
            print_saver.print_and_store(f"Counts ints: {counts_ints}, counts: {counts}")

            max_count = max(counts_ints)
            min_count = min(counts_ints)
            counts_total = max_count - min_count + 1
            if not counts_total == len(counts_ints):
                print_saver.print_and_store(
                    f"Counts total: {counts_total}. Counts ints: {counts_ints}. Min count: {min_count}. Max count: {max_count}. Counts: {counts}. Total steps: {total_steps}."
                )
                repair_log.append(
                    f"Counts total: {counts_total}. Counts ints: {counts_ints}. Min count: {min_count}. Max count: {max_count}. Counts: {counts}. Total steps: {total_steps}."
                )
            # check if total_steps is or isnt equal to the range of counts between the max and min counts
            if total_steps != counts_total:
                # Raise an error, adjust total_steps, or log a warning
                print_saver.print_and_store(
                    f"Total steps {total_steps} is {'less' if total_steps < counts_total else 'greater'} than the total from counts ({counts_total}) by {abs(total_steps - counts_total)}. Min count: {min_count}. Max count: {max_count} with a range of {counts_total}."
                )
                repair_log.append(
                    f"Total steps {total_steps} is {'less' if total_steps < counts_total else 'greater'} than the total from counts ({counts_total}) by {abs(total_steps - counts_total)}. Min count: {min_count}. Max count: {max_count} with a range of {counts_total}."
                )
                total_steps = counts_total if duplicates is not None else total_steps
            print_saver.print_and_store(
                f"Total steps: {total_steps}. Min count: {min_count}. Max count: {max_count}."
            )
            repair_log.append(
                f"Total steps: {total_steps}. Min count: {min_count}. Max count: {max_count}."
            )

        # Step 3: Iterate through each step_number from 1 to total_steps
        steps_temp = steps.copy()
        steps_objs_temp = steps_objs.copy()
        counts, repair_log = self.remove_nonnumeric_counts(
            counts, first_count, repair_log
        )
        for i in range(total_steps):
            step_number = i + 1
            reflection = None
            count_of_i = None
            calc_step_number = None
            for so in steps_objs:
                if so.remaining_budget > 0 and (
                    first_count - (so.remaining_budget + 1) == i + 1
                    or so.step_number == i + 1
                ):
                    count_of_i = so.remaining_budget + 1
                    calc_step_number = i + 1
                    reflection = so.reflection if so.reflection else None
                    print_saver.print_and_store(
                        f"In first attempt to get count_of_i: Step number {step_number}. Count of i: {count_of_i}. Calc step number: {calc_step_number}."
                    )

                    break
            if count_of_i is None:
                count_of_i = (
                    int(counts[i])
                    if i < len(counts) and int(counts[i]) > i - 1
                    else first_count - i
                )
                calc_step_number = first_count - count_of_i + 1
                reflection = reflections[i] if i < len(reflections) else None
                print_saver.print_and_store(
                    f"In second attempt to get count_of_i: Count of i: {count_of_i}. Step number: {step_number}. Calc step number: {calc_step_number}. First count: {first_count}. Steps: {steps}. Steps objs: {steps_objs}. Counts: {counts}."
                )
                repair_log.append(
                    f"In second attempt to get count_of_i: Count of i: {count_of_i}. Step number: {step_number}. Calc step number: {calc_step_number}. First count: {first_count}. Steps: {steps}. Steps objs: {steps_objs}. Counts: {counts}."
                )

            # if reflection is None:
            #     reflection = judge_step(Step(steps[i], step_number, count_of_i), task)
            #     print_saver.print_and_store(
            #         f"Reflection missing for step {step_number}. Generating reflection."
            #     )

            if i == 0:
                # Store original sequence
                original_steps = [
                    (obj.step_number, obj.description) for obj in steps_objs
                ]
                original_numbers = [s[0] for s in original_steps]
                # First pass: identify all issues
                step_issues = {
                    "duplicates": [
                        n
                        for n in set(original_numbers)
                        if original_numbers.count(n) > 1
                    ],
                    "gaps": [
                        (a, b)
                        for a, b in zip(original_numbers[:-1], original_numbers[1:])
                        if b - a > 1
                    ],
                    "inversions": [
                        (a, b)
                        for a, b in zip(original_numbers[:-1], original_numbers[1:])
                        if b <= a
                    ],
                }

                # Log initial analysis
                print_saver.print_and_store(f"Original sequence analysis:")

                for issue_type, issues in step_issues.items():
                    if issues:
                        print_saver.print_and_store(f"- {issue_type.title()}: {issues}")

                # Analyze sequence patterns
                has_gaps = any(
                    b - a > 1
                    for a, b in zip(original_numbers[:-1], original_numbers[1:])
                )
                is_strictly_increasing = all(
                    b > a for a, b in zip(original_numbers[:-1], original_numbers[1:])
                )
                duplicates = [
                    n for n in set(original_numbers) if original_numbers.count(n) > 1
                ]

                # Log diagnostics
                print_saver.print_and_store(f"Original sequence: {original_numbers}")
                print_saver.print_and_store(f"Has gaps: {has_gaps}")
                print_saver.print_and_store(
                    f"Is strictly increasing: {is_strictly_increasing}"
                )
                if duplicates:
                    print_saver.print_and_store(
                        f"Duplicate step numbers found: {duplicates}"
                    )

                # Correct step numbers while logging changes
                for idx, step_obj in enumerate(steps_objs_temp, start=1):
                    if step_obj.step_number != idx:
                        try:
                            original_idx = original_steps.index(
                                (step_obj.step_number, step_obj.description)
                            )
                            gap = step_obj.step_number - idx  # Calculate the gap

                            print_saver.print_and_store(
                                f"Correcting step number for '{step_obj.description}': {step_obj.step_number} -> {idx}. Gap: {gap}. Original index: {original_idx}."
                            )
                            repair_log.append(
                                f"Correcting step number for '{step_obj.description}': {step_obj.step_number} -> {idx}. Gap: {gap}. Original index: {original_idx}."
                            )
                        except ValueError:
                            original_idx = -1  # Indicates not found
                            print_saver.print_and_store(
                                f"Step '{step_obj.description}' with number {step_obj.step_number} not found in original_steps. Assigning new step number {idx}. Gap: {step_obj.step_number - idx}. Original index: {original_idx}. (-1 indicates not found)"
                            )
                        # Update the step number based on several criteria, listed below with logical reasoning for each condition:
                        # If len(steps) == len(steps_objs):
                        #     - Check for gaps in step_number sequence and adjust based on nearest surrounding Step objects.
                        #     - Ensure the order of steps_objs matches the order of steps to maintain consistency.
                        # elif len(steps) > len(steps_objs):
                        #     - Identify missing step descriptions that arent in steps_objs that are in `steps`.
                        #     - Determine if the total gap corresponds to the total number of missing step descriptions, or if the gap corresponds to the difference between the step number of the current step_obj and the step number of the next or previous Step object that is in both lists, and if that gap corresponds to the number of missing steps on either side before reaching a description that is in both lists.
                        #     - Insert new Step objects at appropriate positions based on missing descriptions.
                        #     - Reassign step_number to maintain sequential integrity.
                        #     - steps is the source of truth if it is longer than steps_objs and is equal to total_steps.
                        #     - if gap is equal to the number of missing steps on either side (ie, the gap is due to missing steps_objs entries), adjust step_obj.step_number based on missing steps on either side, as long as the index of the `steps` entries that are missing from `steps_objs` is the same as the missing step number before reaching either the end or the beginning of the list or the previous/next step that is in both lists, in either direction.
                        #     - If the gap is not due to missing steps_objs entries, adjust step_obj.step_number based on the nearest preceding and following Step objects that maintain both the numerical sequence and the original index order.
                        #     - If the order of steps_objs has been compromised, apply a different strategy to reorder the Step objects appropriately.
                        #     - For example, if the idx is less than the step_obj.step_number, lets say idx = 2 and step_obj.step_number = 5, we would expect that the entry at `steps[idx]` would not match the description of the current step_obj so we would need to find the next entry in steps that matches the description of the current step_obj and if the index of that entry plus 1 is equal to the step_obj.step_number, we would know (and should confirm) that if the description of steps_objs[idx] is equal to the description of steps[idx], then we can safely assume that steps[idx:step_obj.step_number] are the missing steps, especially if the description of steps_objs[idx - 1] is equal to the description of steps[idx - 1]. If this is the case, we can safely assume that the missing steps are steps[idx:step_obj.step_number] and we can insert them into steps_objs at the correct indexes.
                        #     - If the idx is greater than the step_obj.step_number, lets say idx = 5 and step_obj.step_number = 2, this might indicate that the order of steps_objs has been compromised and we should check if the description of steps[idx] is equal to the description of the current step_obj. If it is, it likely means the order is messed up, and we need to next determine if each entry by index in steps matches the description of the corresponding entry in steps_objs. If they do, the order is likely correct but the numbers arent, which we can verify by a) checking whether the largest step number in steps_objs is equal to the length of steps_objs (indicating perhaps steps_objs is missing steps from the end of steps, so check if the last few (len(steps) - max(steps_objs.step_number)) steps in steps are missing, otherwise, if the largest step number in steps_objs is equal to the length of `steps`, then we can assume that steps_objs is missing steps from the somewhere in the middle, so we need to verify whether the step numbers are correct by checking if the step number of each step in steps_objs is equal to the index of that step in steps, and if the subset of steps that are in both lists are in the same order.) and b) we can check if the step numbers are correct by checking if the step number of each step in steps_objs is equal to the index of that step in steps, and if not (or in addition to) checking if the subset of steps that are in both lists are in the same order. If they are not, we need to reorder the steps_objs list to match the order of the steps list as long as the length of counts is equal to the length of the steps list and the the original order of counts decreases from on index to the next and never increases or repeats a count. If either the length of counts is not equal to the length of steps or the original order of counts increases from one index to the next, we need to check if the either a) the length of counts is equal to the length of steps_objs and all the counts are in the same order as the original counts or b) the length of counts is equal to the length of steps_objs and the counts are not in the same order as the original counts but the counts correspond exactly to the step numbers of the steps_objs list. If a) is true and b) is not, if the order of step_number values is ascending and sequential, the order is correct and if the number of matching counts to step_numbers subtracted from the length of counts equals the number of missing steps, we should doublecheck that the steps in `steps` that correspond to the indexes of `counts` entries that did not match any step_number values in steps_objs are the missing steps. If this is the case, we can insert them into steps_objs at the correct indexes, using the existing step_number values from the entries in steps_objs that are in the same order as the missing steps in steps. If b) is true, we need to check if the counts correspond to the step numbers of the steps_objs list. If they do and the step_numbers only increase from one index to the next, we can assume the order is correct and the counts are correct and we can assume that the missing steps are the steps in `steps` that correspond to the indexes of `counts` entries that did not match any step_number values in steps_objs if the number of missing string descriptions that are in `steps` but not in `[step_obj.description for step_obj in steps_objs]`. If this is the case, we can insert them into steps_objs at the correct indexes, using the existing step_number values from the entries in steps_objs that are in the same order as the missing steps in steps. If the counts do not correspond to the step numbers of the steps_objs list, we need to check if the counts are in the same order as the original counts and if the length of counts is equal to the length of steps_objs. If both are true, we can assume that the order is correct and the counts are correct and we can assume that the missing steps are the steps in `steps` that correspond to the indexes of `counts` entries that did not match any step_number values in steps_objs. If this is the case, we can insert them into steps_objs at the correct indexes, using the existing step_number values from the entries in steps_objs that are in the same order as the missing steps in steps.
                        # elif len(steps) < len(steps_objs):
                        #     - Identify missing step descriptions that arent in `steps` list that are in `steps_objs`.
                        #     - Determine if the total gap corresponds to the total number of missing step descriptions, or if the gap corresponds to the difference between the step number of the current step_obj and the step number of the next or previous Step object that is in both lists, and if that gap corresponds to the number of missing steps on either side before reaching a description that is in both lists.
                        #     - Insert new Step objects at appropriate positions based on missing descriptions.
                        #     - Reassign step_number to maintain sequential integrity.
                        #     - steps_objs is the source of truth if it is longer than steps and is equal to total_steps.
                        #     - Remembering that steps_objs is longer than the `steps` list, if the gap is equal to the number of missing steps on either side (ie, the gap is due to missing steps entries), if steps_objs is the source of truth because it is longer than the `steps` list, we can assume that the missing steps are steps in `steps_objs` that are not in `steps` and we can insert them into steps at the correct indexes, as long as the step_number values of steps_objs are in ascending order and sequential. If the step_number values are not in ascending order and sequential, we need to check if the counts are in ascending order and sequential and if the counts are in ascending order and sequential, we can assume that the missing steps are the steps in `steps_objs` that are not in `steps` and we can insert them into steps at the correct indexes, using the existing step_number values from the entries in steps_objs that are in the same order as the missing steps in steps. If the counts are not in ascending order and sequential, we need to check if the counts correspond to the step numbers of the steps_objs list. If they do and the step_numbers only increase from one index to the next, we can assume the order is correct and the counts are correct and we can assume that the missing steps are the steps in `steps` that correspond to the indexes of `counts` entries that did not match any step_number values in steps_objs. If this is the case, we can insert them into steps_objs at the correct indexes, using the existing step_number values from the entries in steps_objs that are in the same order as the missing steps in steps. If the counts do not correspond to the step numbers of the steps_objs list, we need to check if the counts are in the same order as the original counts and if the length of counts is equal to the length of steps_objs. If both are true, we can assume that the order is correct and the counts are correct and we can assume that the missing steps are the steps in `steps` that correspond to the indexes of `counts` entries that did not match any step_number values in steps_objs. If this is the case, we can insert them into steps_objs at the correct indexes, using the existing step_number values from the entries in steps_objs that are in the same order as the missing steps in steps, as long as each subset of step descriptions that are in both lists are in the same order as each other. If they are not, based on a) which list is longer and b) whether the step_number values are in ascending order and sequential and c) whether the counts are in ascending order and sequential and d) whether the counts correspond to the step numbers of the steps_objs list, we can determine which list is the source of truth and insert the missing steps into the other list at the correct indexes.
                        #     - If the gap is not due to missing steps entries, adjust step_obj.step_number based on the nearest preceding and following Step objects that maintain both the numerical sequence and the original index order.
                        #     - If the order of steps_objs has been compromised, apply a different strategy to reorder the Step objects appropriately.
                        #     - For example, if the idx is less than the step_obj.step_number, lets say idx = 2 and step_obj.step_number = 5, we would expect that the entry at `steps[idx]` would not match the description of the current step_obj so we would need to find the next entry in steps that matches the description of the current step_obj and if the index of that entry plus 1 is equal to the step_obj.step_number, we would know (and should confirm) that if the description of steps_objs[idx] is equal to the description of steps[idx], then we can safely assume that steps[idx:step_obj.step_number] are the missing steps, especially if the description of steps_objs[idx - 1] is equal to the description of steps[idx - 1]. If this is the case, we can safely assume that the missing steps are steps[idx:step_obj.step_number] and we can insert them into steps_objs at the correct indexes.
                        #     - If the idx is greater than the step_obj.step_number, lets say idx = 5 and step_obj.step_number = 2, this might indicate that the order of steps_objs has been compromised and we should check if the description of steps[idx] is equal to the description of the current step_obj. If it is, it likely means the order is messed up, and we need to next determine if each entry by index in steps matches the description of the corresponding entry in steps_objs. If they do, the order is likely correct but the numbers arent, which we can verify by a) checking whether the largest step number in steps_objs is equal to the length of steps_objs (indicating perhaps steps_objs is missing steps from the end of steps, so check if the last few (len(steps) - max(steps_objs.step_number)) steps in steps are missing, otherwise, if the largest step number in steps_objs is equal to the length of `steps`, then we can assume that steps_objs is missing steps from the somewhere in the middle, so we need to verify whether the step numbers are correct by checking if the step number of each step in steps_objs is equal to the index of that step in steps, and if the subset of steps that are in both lists are in the same order.) and b) we can check if the step numbers are correct by checking if the step number of each step in steps_objs is equal to the index of that step in steps, and if not (or in addition to) checking if the subset of steps that are in both lists are in the same order. If they are not, we need to reorder the steps_objs list to match the order of the steps list as long as the length of counts is equal to the length of the steps list and the the original order of counts decreases from on index to the next and never increases or repeats a count. If either the length of counts is not equal to the length of steps or the original order of counts increases from one index to the next, we need to check if the either a) the length of counts is equal to the length of steps_objs and all the counts are in the same order as the original counts or b) the length of counts is equal to the length of steps_objs and the counts are not in the same order as the original counts but the counts correspond exactly to the step numbers of the steps_objs list. If a) is true and b) is not, if the order of step_number values is ascending and sequential, the order is correct and if the number of matching counts to step_numbers subtracted from the length of counts equals the number of missing steps, we should doublecheck that the steps in `steps` that correspond to the indexes of `counts` entries that did not match any step_number values in steps_objs are the missing steps. If this is the case, we can insert them into steps_objs at the correct indexes, using the existing step_number values from the entries in steps_objs that are in the same order as the missing steps in steps. If b) is true, we need to check if the counts correspond to the step numbers of the steps_objs list. If they do and the step_numbers only increase from one index to the next, we can assume the order is correct and the counts are correct and we can assume that the missing steps are the steps in `steps` that correspond to the indexes of `counts` entries that did not match any step_number values in steps_objs if the number of missing string descriptions that are in `steps` but not in `[step_obj.description for step_obj in steps_objs]`. If this is the case, we can insert them into steps_objs at the correct indexes, using the existing step_number values from the entries in steps_objs that are in the same order as the missing steps in steps. If the counts do not correspond to the step numbers of the steps_objs list, we need to check if the counts are in the same order as the original counts and if the length of counts is equal to the length of steps_objs. If both are true, we can assume that the order is correct and the counts are correct and we can assume that the missing steps are the steps in `steps` that correspond to the indexes of `counts` entries that did not match any step_number values in steps_objs. If this is the case, we can insert them into steps_objs at the correct indexes, using the existing step_number values from the entries in steps_objs that are in the same order as the missing steps in steps.
                        # Check for order discrepancies and apply alternative logic if the order has been compromised.
                        step_obj.step_number = idx

                # Re-analyze after correction
                corrected_numbers = [obj.step_number for obj in steps_objs_temp]
                has_gaps = any(
                    b - a > 1
                    for a, b in zip(corrected_numbers[:-1], corrected_numbers[1:])
                )
                is_strictly_increasing = all(
                    b > a for a, b in zip(corrected_numbers[:-1], corrected_numbers[1:])
                )
                duplicates = [
                    n for n in set(corrected_numbers) if corrected_numbers.count(n) > 1
                ]

                # Log post-correction diagnostics
                print_saver.print_and_store(f"Corrected sequence: {corrected_numbers}")
                repair_log.append(f"Corrected sequence: {corrected_numbers}")
                print_saver.print_and_store(f"Has gaps after correction: {has_gaps}")
                repair_log.append(f"Has gaps after correction: {has_gaps}")
                print_saver.print_and_store(
                    f"Is strictly increasing after correction: {is_strictly_increasing}"
                )
                repair_log.append(
                    f"Is strictly increasing after correction: {is_strictly_increasing}"
                )

                if duplicates:
                    for duplicate in duplicates:
                        print_saver.print_and_store(
                            f"Handling duplicate step number: {duplicate}"
                        )
                        # Example strategy: Increment step numbers of subsequent steps
                        for step_obj in steps_objs:
                            if step_obj.step_number == duplicate:
                                step_obj.step_number += 1
                                print_saver.print_and_store(
                                    f"Incremented step number for '{step_obj.description}' to {step_obj.step_number}."
                                )
                                repair_log.append(
                                    f"Incremented step number for '{step_obj.description}' to {step_obj.step_number}."
                                )
            if len(steps) < total_steps and total_steps == len(steps_objs):
                print_saver.print_and_store(
                    f"Index {i}: Steps length mismatch (less in steps than in steps_objs, and len of steps_objs is equal to total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                )
                repair_log.append(
                    f"Index {i}: Steps length mismatch (less in steps than in steps_objs, and len of steps_objs is equal to total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}. Steps: {steps}. Steps objs: {steps_objs}."
                )
                missing_objs = [
                    obj for obj in steps_objs if obj.description not in steps
                ]
                for idx, obj in enumerate(missing_objs):
                    # Find the next step that appears in both lists
                    next_match = None
                    obj_index = steps_objs.index(obj)
                    for following in steps_objs[obj_index + 1 :]:
                        if following.description in steps:
                            next_match = following.description
                            break
                    # If there's a subsequent match, insert right before it; otherwise append
                    if next_match:
                        insert_idx = steps.index(next_match)
                    else:
                        insert_idx = len(steps)
                    steps_temp.insert(insert_idx, obj.description)
                repair_log.append(
                    f"Missing objects added to steps: {missing_objs}. Steps: {steps_temp}."
                )
            elif len(steps) > total_steps and total_steps == len(steps_objs):
                print_saver.print_and_store(
                    f"Index {i}: Steps length mismatch (more in steps than in steps_objs, and len of steps_objs is equal to total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                )
                repair_log.append(
                    f"Index {i}: Steps length mismatch (more in steps than in steps_objs, and len of steps_objs is equal to total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}. Steps: {steps}. Steps objs: {steps_objs}."
                )
                missing_steps = [
                    step
                    for step in steps
                    if step not in [obj.description for obj in steps_objs]
                ]
                for idx, step in enumerate(missing_steps):
                    # Find the next step that appears in both lists
                    next_match = None
                    step_index = steps.index(step)
                    count = int(counts[step_index]) if step_index < len(counts) else 0
                    # calc_step_number = first_count - count + 1
                    for following in steps[step_index + 1 :]:
                        if following in [obj.description for obj in steps_objs]:
                            next_match = following
                            break
                    # If there's a subsequent match, insert right before it; otherwise append
                    if next_match:
                        insert_idx = [obj.description for obj in steps_objs].index(
                            next_match
                        )
                        # if insert_idx < calc_step_number - 1:
                        #     insert_idx = calc_step_number - 1
                    else:
                        insert_idx = calc_step_number
                    # if (
                    #     steps_objs[insert_idx].step_number == calc_step_number
                    #     and steps_objs[insert_idx].step_number == 1
                    # ):
                    #     insert_idx += 1
                    if insert_idx < len(counts):
                        steps_objs_temp.insert(
                            insert_idx,
                            Step(
                                step,
                                insert_idx + 1,
                                int(counts[insert_idx]) - 1,
                                reflection,
                            ),
                        )
                    else:
                        steps_objs_temp.append(
                            Step(
                                step,
                                insert_idx + 1,
                                first_count - insert_idx - 1,
                                reflection,
                            )
                        )

                repair_log.append(
                    f"Missing steps added to steps_objs: {missing_steps}. Steps objs: {steps_objs_temp}."
                )
            elif len(steps) < total_steps and total_steps < len(steps_objs):
                print_saver.print_and_store(
                    f"Index {i}: Steps length mismatch (less in steps than in steps_objs, and len of steps_objs is greater than total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                )
                repair_log.append(
                    f"Index {i}: Steps length mismatch (less in steps than in steps_objs, and len of steps_objs is greater than total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}. Steps: {steps}. Steps objs: {steps_objs}."
                )
                missing_objs = [
                    obj for obj in steps_objs if obj.description not in steps
                ]
                for idx, obj in enumerate(missing_objs):
                    # Find the next step that appears in both lists
                    next_match = None
                    obj_index = steps_objs.index(obj)
                    for following in steps_objs[obj_index + 1 :]:
                        if following.description in steps:
                            next_match = following.description
                            break
                    # If there's a subsequent match, insert right before it; otherwise append
                    if next_match:
                        insert_idx = steps.index(next_match)
                    else:
                        insert_idx = len(steps)
                    steps_temp.insert(insert_idx, obj.description)
                repair_log.append(
                    f"Missing objects added to steps: {missing_objs}. Steps: {steps_temp}."
                )
            elif len(steps) > total_steps and total_steps < len(steps_objs):
                print_saver.print_and_store(
                    f"Index {i}: Steps length mismatch (more in steps than in steps_objs, and len of steps_objs is greater than total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                )
                repair_log.append(
                    f"Index {i}: Steps length mismatch (more in steps than in steps_objs, and len of steps_objs is greater than total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}. Steps: {steps}. Steps objs: {steps_objs}."
                )
                missing_steps = [
                    step
                    for step in steps
                    if step not in [obj.description for obj in steps_objs]
                ]
                for idx, step in enumerate(missing_steps):
                    # Find the next step that appears in both lists
                    next_match = None
                    step_index = steps.index(step)
                    for following in steps[step_index + 1 :]:
                        if following in [obj.description for obj in steps_objs]:
                            next_match = following
                            break
                    # If there's a subsequent match, insert right before it; otherwise append
                    if next_match:
                        insert_idx = [obj.description for obj in steps_objs].index(
                            next_match
                        )
                    else:
                        insert_idx = (
                            calc_step_number
                            if calc_step_number <= len(steps_objs)
                            else len(steps_objs)
                        )
                    if insert_idx < len(counts):
                        steps_objs_temp.insert(
                            insert_idx,
                            Step(
                                step,
                                insert_idx + 1,
                                int(counts[insert_idx]) - 1,
                                reflection,
                            ),
                        )
                    else:
                        steps_objs_temp.append(
                            Step(
                                step,
                                insert_idx + 1,
                                first_count - insert_idx - 1,
                                reflection,
                            ),
                        )
                repair_log.append(
                    f"Missing steps added to steps_objs: {missing_steps}. Steps objs: {steps_objs_temp}."
                )
            elif len(steps) == total_steps and total_steps < len(steps_objs):
                print_saver.print_and_store(
                    f"Index {i}: Steps length mismatch (equal in steps and steps_objs, and len of steps_objs is greater than total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                )
                repair_log.append(
                    f"Index {i}: Steps length mismatch (equal in steps and steps_objs, and len of steps_objs is greater than total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}. Steps: {steps}. Steps objs: {steps_objs}."
                )
                missing_objs = [
                    obj for obj in steps_objs if obj.description not in steps
                ]
                for idx, obj in enumerate(missing_objs):
                    # Find the next step that appears in both lists
                    next_match = None
                    obj_index = steps_objs.index(obj)
                    for following in steps_objs[obj_index + 1 :]:
                        if following.description in steps:
                            next_match = following.description
                            break
                    # If there's a subsequent match, insert right before it; otherwise append
                    if next_match:
                        insert_idx = steps.index(next_match)
                    else:
                        insert_idx = len(steps)
                    steps_temp.insert(insert_idx, obj.description)
                repair_log.append(
                    f"Missing objects added to steps: {missing_objs}. Steps: {steps_temp}."
                )
            elif len(steps) == total_steps and total_steps > len(steps_objs):
                print_saver.print_and_store(
                    f"Index {i}: Steps length mismatch (equal in steps and steps_objs, and len of steps_objs is less than total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                )
                repair_log.append(
                    f"Index {i}: Steps length mismatch (equal in steps and steps_objs, and len of steps_objs is less than total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}. Steps: {steps}. Steps objs: {steps_objs}."
                )
                missing_steps = [
                    step
                    for step in steps
                    if step not in [obj.description for obj in steps_objs]
                ]
                for idx, step in enumerate(missing_steps):
                    # Find the next step that appears in both lists
                    next_match = None
                    step_index = steps.index(step)
                    for following in steps[step_index + 1 :]:
                        if following in [obj.description for obj in steps_objs]:
                            next_match = following
                            break
                    # If there's a subsequent match, insert right before it; otherwise append
                    if next_match:
                        insert_idx = [obj.description for obj in steps_objs].index(
                            next_match
                        )
                    else:
                        insert_idx = (
                            calc_step_number
                            if calc_step_number <= len(steps_objs)
                            else len(steps_objs)
                        )
                    if insert_idx < len(counts) and len(steps_objs) != 0:
                        print_saver.print_and_store(
                            f"Inserting step '{step}' at index {insert_idx} of counts: {counts} of length {len(counts)}."
                        )
                        print_saver.print_and_store(
                            f"counts[insert_idx]: {counts[insert_idx]}."
                        )
                        steps_objs_temp.insert(
                            insert_idx,
                            Step(
                                step,
                                insert_idx + 1,
                                int(counts[insert_idx]) - 1,
                                reflection,
                            ),
                        )
                    else:
                        steps_objs_temp.append(
                            Step(
                                step,
                                insert_idx + 1,
                                first_count - insert_idx - 1,
                                reflection,
                            ),
                        )
                repair_log.append(
                    f"Missing steps added to steps_objs: {missing_steps}. Steps objs: {steps_objs_temp}."
                )
            elif len(steps) < total_steps and total_steps > len(steps_objs):
                # This result indicates that the steps list is shorter than the total steps and the steps_objs list is shorter than the total steps. So we should check if the steps list is shorter than the steps_objs list or vice versa as the total steps is greater than both. Even if they are of the same length, we need to check if either has steps not in the other, as each could have unique steps, and adding them together might result in both lists equaling the total steps, if we're lucky.
                # This result indicates that the steps list is shorter than the total steps and the steps_objs list is also shorter than the total steps.
                # We should check if one list is shorter than the other or if they are the same length, then handle missing items in each accordingly.

                if len(steps) < len(steps_objs):
                    print_saver.print_and_store(
                        f"Index {i}: Steps length mismatch (less in steps than in steps_objs, total_steps is higher). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                    )
                    repair_log.append(
                        f"Index {i}: Steps length mismatch (less in steps than in steps_objs, total_steps is higher). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}. Steps: {steps}. Steps objs: {steps_objs}."
                    )
                    missing_objs = [
                        obj for obj in steps_objs if obj.description not in steps
                    ]
                    for idx, obj in enumerate(missing_objs):
                        next_match = None
                        obj_index = steps_objs.index(obj)
                        for following in steps_objs[obj_index + 1 :]:
                            if following.description in steps:
                                next_match = following.description
                                break
                        if next_match:
                            insert_idx = steps.index(next_match)
                        else:
                            insert_idx = len(steps)
                        steps_temp.insert(insert_idx, obj.description)
                    repair_log.append(
                        f"Missing objects added to steps: {missing_objs}. Steps: {steps_temp}."
                    )
                elif len(steps) > len(steps_objs):
                    print_saver.print_and_store(
                        f"Index {i}: Steps length mismatch (more in steps than in steps_objs, total_steps is higher). Adjusting steps_objs. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                    )
                    repair_log.append(
                        f"Index {i}: Steps length mismatch (more in steps than in steps_objs, total_steps is higher). Adjusting steps_objs. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}. Steps: {steps}. Steps objs: {steps_objs}."
                    )
                    missing_steps = [
                        step
                        for step in steps
                        if step not in [obj.description for obj in steps_objs]
                    ]
                    for idx, step_value in enumerate(missing_steps):
                        next_match = None
                        step_index = steps.index(step_value)
                        for following in steps[step_index + 1 :]:
                            if following in [obj.description for obj in steps_objs]:
                                next_match = following
                                break
                        if next_match:
                            insert_idx = [obj.description for obj in steps_objs].index(
                                next_match
                            )
                        else:
                            insert_idx = (
                                calc_step_number
                                if calc_step_number <= len(steps_objs)
                                else len(steps_objs)
                            )
                        steps_objs_temp.insert(
                            insert_idx,
                            Step(
                                step_value,
                                insert_idx + 1,
                                (
                                    int(counts[insert_idx]) - 1
                                    if insert_idx < len(counts)
                                    else 0
                                ),
                                reflection,
                            ),
                        )
                    repair_log.append(
                        f"Missing steps added to steps_objs: {missing_steps}. Steps objs: {steps_objs_temp}."
                    )
                elif len(steps) == len(steps_objs):
                    # len(steps) == len(steps_objs), but both are still less than total_steps.
                    # We try reconciling any missing content in each list to move toward total_steps.
                    print_saver.print_and_store(
                        f"Index {i}: Steps length mismatch (equal in steps and steps_objs, but both less than total_steps). Adjusting. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                    )
                    repair_log.append(
                        f"Index {i}: Steps length mismatch (equal in steps and steps_objs, but both less than total_steps). Adjusting. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}. Steps: {steps}. Steps objs: {steps_objs}."
                    )
                    missing_objs = [
                        obj for obj in steps_objs if obj.description not in steps
                    ]

                    if calc_step_number is None:  # This should never happen
                        print_saver.print_and_store(
                            f"Index {i}: Error calculating step number for {i}."
                        )
                        continue

                    for obj in missing_objs:
                        obj_index = steps_objs.index(obj)
                        next_match = None
                        for following in steps_objs[obj_index + 1 :]:
                            if following.description in steps:
                                next_match = following.description
                                break
                        if next_match:
                            insert_idx = steps.index(next_match)
                        else:
                            insert_idx = calc_step_number
                        steps_temp.insert(insert_idx, obj.description)

                    missing_steps = [
                        step_value
                        for step_value in steps
                        if step_value not in [o.description for o in steps_objs]
                    ]
                    for step_value in missing_steps:
                        step_index = steps.index(step_value)
                        next_match = None
                        for following in steps[step_index + 1 :]:
                            if following in [o.description for o in steps_objs]:
                                next_match = following
                                break
                        if next_match:
                            insert_idx = [o.description for o in steps_objs].index(
                                next_match
                            )
                        else:
                            print_saver.print_and_store(
                                f"Index {i}: Error calculating step number for '{step_value}' so using calc_step_number {calc_step_number}."
                            )
                            insert_idx = calc_step_number
                        steps_objs_temp.insert(
                            insert_idx,
                            Step(
                                step_value,
                                insert_idx + 1,
                                (
                                    int(counts[insert_idx]) - 1
                                    if insert_idx < len(counts)
                                    else 0
                                ),
                                reflection,
                            ),
                        )
                    repair_log.append(
                        f"Missing objects added to steps: {missing_objs}. Steps: {steps_temp}."
                    )

            elif len(steps) > total_steps and total_steps > len(steps_objs):
                # This condition means that the steps list is longer than the total steps and the steps_objs list is also shorter than the total steps, meaning that the steps list is longer than the steps_objs list, but somehow the total steps is greater than both. #
                # This is a tricky situation, but we can handle it by adding the missing steps from the steps list to the steps_objs list.
                print_saver.print_and_store(
                    f"Index {i}: Steps length mismatch (more in steps than in steps_objs, and total_steps is in between). Adjusting steps_objs. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                )
                repair_log.append(
                    f"Index {i}: Steps length mismatch (more in steps than in steps_objs, and total_steps is in between). Adjusting steps_objs. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}. Steps: {steps}. Steps objs: {steps_objs}."
                )
                missing_steps = [
                    step
                    for step in steps
                    if step not in [obj.description for obj in steps_objs]
                ]
                for idx, step in enumerate(missing_steps):
                    next_match = None
                    step_index = steps.index(step)
                    for following in steps[step_index + 1 :]:
                        if following in [obj.description for obj in steps_objs]:
                            next_match = following
                            break
                    if next_match:
                        insert_idx = [obj.description for obj in steps_objs].index(
                            next_match
                        )
                    else:
                        insert_idx = (
                            calc_step_number
                            if calc_step_number <= len(steps_objs)
                            else len(steps_objs)
                        )
                    if insert_idx < len(counts):
                        steps_objs_temp.insert(
                            insert_idx,
                            Step(
                                step,
                                insert_idx + 1,
                                int(counts[insert_idx]) - 1,
                                reflection,
                            ),
                        )
                    else:
                        steps_objs_temp.append(
                            Step(
                                step,
                                insert_idx + 1,
                                first_count - insert_idx - 1,
                                reflection,
                            ),
                        )
                    print_saver.print_and_store(
                        f"Inserted missing step '{step}' at index {insert_idx}."
                    )
                    repair_log.append(
                        f"Inserted missing step '{step}' at index {insert_idx}."
                    )
                unique_step_descriptions = set(s.strip() for s in steps if s.strip())
                unique_obj_descriptions = set(
                    obj.description.strip()
                    for obj in steps_objs
                    if obj.description.strip()
                )
                total_steps = len(
                    unique_step_descriptions.union(unique_obj_descriptions)
                )
                print_saver.print_and_store(
                    f"Final total steps after adjustment: {total_steps}"
                )
                repair_log.append(
                    f"Final total steps after adjustment: {total_steps}. Unique steps: {unique_step_descriptions}. Unique objects: {unique_obj_descriptions}."
                )
            elif len(steps) == len(steps_objs) and total_steps < max(
                len(steps), len(steps_objs)
            ):
                print_saver.print_and_store(
                    f"Index {i}: Steps length mismatch (equal in steps and steps_objs, but both greater than total_steps). Adjusting. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                )
                repair_log.append(
                    f"Index {i}: Steps length mismatch (equal in steps and steps_objs, but both greater than total_steps). Adjusting. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}. Steps: {steps}. Steps objs: {steps_objs}."
                )
                missing_objs = [
                    obj for obj in steps_objs if obj.description not in steps
                ]
                for obj in missing_objs:
                    obj_index = steps_objs.index(obj)
                    next_match = None
                    for following in steps_objs[obj_index + 1 :]:
                        if following.description in steps:
                            next_match = following.description
                            break
                    if next_match:
                        insert_idx = steps.index(next_match)
                    else:
                        insert_idx = len(steps)
                    steps_temp.insert(insert_idx, obj.description)
                repair_log.append(
                    f"Missing objects added to steps: {missing_objs}. Steps: {steps_temp}."
                )
                missing_steps = [
                    step_value
                    for step_value in steps
                    if step_value not in [o.description for o in steps_objs]
                ]
                for step_value in missing_steps:
                    step_index = steps.index(step_value)
                    next_match = None
                    for following in steps[step_index + 1 :]:
                        if following in [o.description for o in steps_objs]:
                            next_match = following
                            break
                    if next_match:
                        insert_idx = [o.description for o in steps_objs].index(
                            next_match
                        )
                    else:
                        insert_idx = (
                            calc_step_number
                            if calc_step_number <= len(steps_objs)
                            else len(steps_objs)
                        )
                    steps_objs_temp.insert(
                        insert_idx,
                        Step(
                            step_value,
                            insert_idx + 1,
                            (
                                int(counts[insert_idx]) - 1
                                if insert_idx < len(counts)
                                else 0
                            ),
                            reflection,
                        ),
                    )
                repair_log.append(
                    f"Missing steps added to steps_objs: {missing_steps}. Steps objs: {steps_objs_temp}."
                )
            elif len(steps) == len(steps_objs) and total_steps > max(
                len(steps), len(steps_objs)
            ):
                print_saver.print_and_store(
                    f"Index {i}: Steps length mismatch (equal in steps and steps_objs, but both less than total_steps). Adjusting. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                )
                repair_log.append(
                    f"Index {i}: Steps length mismatch (equal in steps and steps_objs, but both less than total_steps). Adjusting. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}. Steps: {steps}. Steps objs: {steps_objs}."
                )
                missing_objs = [
                    obj for obj in steps_objs if obj.description not in steps
                ]
                for obj in missing_objs:
                    obj_index = steps_objs.index(obj)
                    next_match = None
                    for following in steps_objs[obj_index + 1 :]:
                        if following.description in steps:
                            next_match = following.description
                            break
                    if next_match:
                        insert_idx = steps.index(next_match)
                    else:
                        insert_idx = len(steps)
                    steps_temp.insert(insert_idx, obj.description)
                repair_log.append(
                    f"Missing objects added to steps: {missing_objs}. Steps: {steps_temp}."
                )

                missing_steps = [
                    step_value
                    for step_value in steps
                    if step_value not in [o.description for o in steps_objs]
                ]
                for step_value in missing_steps:
                    step_index = steps.index(step_value)
                    next_match = None
                    for following in steps[step_index + 1 :]:
                        if following in [o.description for o in steps_objs]:
                            next_match = following
                            break
                    if next_match:
                        insert_idx = [o.description for o in steps_objs].index(
                            next_match
                        )
                    else:
                        insert_idx = (
                            calc_step_number
                            if calc_step_number <= len(steps_objs)
                            else len(steps_objs)
                        )
                    steps_objs_temp.insert(
                        insert_idx,
                        Step(
                            step_value,
                            insert_idx + 1,
                            (
                                int(counts[insert_idx]) - 1
                                if insert_idx < len(counts)
                                else 0
                            ),
                            reflection,
                        ),
                    )
                repair_log.append(
                    f"Missing steps added to steps_objs: {missing_steps}. Steps objs: {steps_objs_temp}."
                )
            elif len(steps) == total_steps and total_steps == len(steps_objs):
                # This condition means that the steps list is equal to the total steps and the steps_objs list is also equal to the total steps, meaning that the steps list is equal to the steps_objs list, and both are equal to the total steps.
                # This is the ideal situation, and we don't need to do anything.
                print_saver.print_and_store(
                    f"Index {i}: Steps length match (equal in steps and steps_objs, and equal to total_steps). No adjustment needed. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                )

            else:
                raise ValueError(
                    f"Index {i}: Unhandled case. Please check the length of steps, steps_objs, and total_steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                )
            steps = steps_temp
            steps_objs = steps_objs_temp

            # counts = remove_nonnumeric_counts(counts, first_count)
            # pre_count = (
            #     counts[i] if i < len(counts) else (int(counts[-1]) - 1 if counts else 0)
            # )
            # pre_count = int(pre_count)
            remaining_budget = count_of_i - 1 if count_of_i is not (None or 0) else 0
            description = None
            for j in range(len(steps_objs)):
                if (
                    steps_objs[j].step_number == step_number
                    and steps_objs[j].remaining_budget == remaining_budget
                ) or (
                    steps_objs[j].step_number == calc_step_number
                    and steps_objs[j].remaining_budget == remaining_budget
                ):
                    description = steps_objs[j].description
                    break
            if description is None:
                try:
                    count_index = int(counts_ints.index(count_of_i))
                    description = (
                        steps[count_index] if count_index < len(steps) else "ERROR"
                    )
                except Exception as e:
                    description = "ERROR " + str(e)
                    print_saver.print_and_store(
                        f"Index {i}: Error finding description for step {step_number}: error {e}"
                    )
                if "ERROR" in description:
                    try:
                        description = steps[i]
                    except Exception as e:
                        description = "ERROR " + str(e)
                        print_saver.print_and_store(
                            f"Index {i}: Error finding description for step {step_number}: error {e}"
                        )
            assert description is not None, f"Description is None for index {i}."
            ee = description.replace("ERROR", "") if "ERROR" in description else ""
            assert (
                "ERROR" not in description
            ), f"Description is 'ERROR' for index {i}. Error:{ee} Steps: {steps}. Counts: {counts_ints}. Counts type: {type(counts_ints[0])} Steps_objs: {steps_objs} and remaining_budget: {remaining_budget} and count_of_i: {count_of_i} and calc_step_number: {calc_step_number}"
            print_saver.print_and_store(
                f"Index {i}: Step {step_number} description: {description}, remaining_budget: {remaining_budget}"
            )
            # assert (
            #     step_number == calc_step_number
            # ), f"Step number mismatch: {step_number} vs {calc_step_number} for index {i} step {description}. Steps: {steps}. Counts: {counts}. Steps_objs: {steps_objs}."
            if step_number in existing_map.keys():
                obj = existing_map[step_number]
                # Update description if necessary
                if description and obj.description != description:
                    obj.description = description
                # Update remaining_budget if necessary
                if remaining_budget != obj.remaining_budget:
                    if remaining_budget < obj.remaining_budget:
                        print_saver.print_and_store(
                            f"Remaining budget mismatch for step {step_number} (shorter than expected). Adjusting remaining budget. {remaining_budget} vs {obj.remaining_budget}"
                        )
                        remaining_budget = obj.remaining_budget
                        # obj.remaining_budget = remaining_budget
                    elif remaining_budget > obj.remaining_budget:
                        print_saver.print_and_store(
                            f"Remaining budget mismatch for step {step_number} (longer than expected). Adjusting remaining budget. {remaining_budget} vs {obj.remaining_budget}"
                        )
                        obj.remaining_budget = remaining_budget
            else:
                # Create a new Step object
                print_saver.print_and_store(
                    f"Creating new Step object for step {step_number}: {description}"
                )
                new_obj = Step(description, step_number, remaining_budget, reflection)
                existing_map[step_number] = new_obj

        # Step 4: Sort the steps by step_number
        step_objs_final = [existing_map[sn] for sn in sorted(existing_map.keys())]

        return step_objs_final, steps, reflections, repair_log

    # ===========================
    # Define the process_steps Function
    # ===========================

    def process_steps(
        self,
        steps_objs: Optional[List[Step]],
        steps: List[str],
        counts: List[str | int],
        reflections: List[Optional[Reflection]],
        rewards: List[float],
        response: str,
        first_count: int,
        print_saver,  # Assuming print_saver is an object with print_and_store method
        interaction: Interaction,  # Assuming Interaction is a defined class
        task: str,
        current_plan_step: int,
        repair_log: Optional[List[str]] = None,
    ) -> Interaction:
        """
        Process and consolidate steps, reflections, and rewards into the interaction object.

        Args:
            steps_objs (Optional[List[Step]]): Existing list of Step objects.
            steps (List[str]): List of step descriptions.
            counts (List[str | int]): List of counts representing the budget before each step.
            reflections (List[Optional[Reflection | str]]): List of reflections.
            rewards (List[float]): List of rewards.
            response (str): Response string containing XML-like tags.
            first_count (int): The initial count value.
            print_saver: Object with a print_and_store method.
            interaction (Interaction): Object to store steps and reflections.
            task (str): The current task context.

        Returns:
            Interaction: The updated interaction object.
        """
        # Ensure steps_objs is a list
        steps_objs = steps_objs or []

        # ===========================
        # Step 1: Consolidate Steps
        # ===========================

        # Call the consolidate_steps function to align steps_objs with steps and counts
        steps_objs, steps, reflections, repair_log = self.consolidate_steps(
            steps_objs,
            steps,
            counts,
            reflections,
            first_count,
            response,
            current_plan_step,
            repair_log,
        )

        assert 0 not in [
            obj.step_number for obj in steps_objs
        ], "Step numbers must be > 0."

        # ===========================
        # Step 2: Validate and Adjust Steps
        # ===========================

        repair_log = self.validate_steps(
            steps_objs,
            steps,
            counts,
            reflections,
            rewards,
            print_saver,
            interaction,
            task,
            first_count,
            repair_log,
        )

        # ===========================
        # Step 3: Ensure Content Similarity
        # ===========================

        repair_log = self.ensure_content_similarity(
            steps_objs, steps, first_count, print_saver, interaction, task, repair_log
        )

        # ===========================
        # Step 4: Update Interaction with Consolidated Steps
        # ===========================

        for step_obj in steps_objs:
            if step_obj not in interaction.steps:
                interaction.steps.append(step_obj)

        # ===========================
        # Step 5: Process Reflections and Rewards
        # ===========================

        for i, step_obj in [(obj.step_number, obj) for obj in steps_objs]:
            # Handle reflections
            if i == 0 and step_obj.step_number == 0:
                step_obj.step_number = 1
                # Now check steps after the first one to ensure they are sequential
                for j in range(1, len(steps_objs)):
                    if steps_objs[j].step_number != step_obj.step_number + 1:
                        steps_objs[j].step_number = step_obj.step_number + 1
            if (
                step_obj.reflection is None
                or not isinstance(step_obj.reflection, Reflection)
                or step_obj.reflection.content.strip() == ""
            ) and step_obj.step_number in [r.step_number for r in reflections]:
                reflection = (
                    reflections[
                        reflections.index(
                            [
                                r
                                for r in reflections
                                if r.step_number == step_obj.step_number
                            ][0]
                        )
                    ]
                    if step_obj.step_number in [r.step_number for r in reflections]
                    else self.judge_step(step_obj, task)
                )
                if reflection is not None and isinstance(reflection, Reflection):
                    reflection.step_number = step_obj.step_number
                    step_obj.reflection = reflection
                    interaction.reflections.append(reflection)
                elif isinstance(reflection, str):
                    step_obj.reflection = (
                        reflection
                        if isinstance(reflection, Reflection)
                        else Reflection(
                            content=str(reflection),
                            reward=(
                                float(rewards[i])
                                if i < len(rewards)
                                else self.judge_step(step_obj, task)
                            ),
                            step_number=step_obj.step_number,
                        )
                    )
            if step_obj.reflection is None or not isinstance(
                step_obj.reflection, Reflection
            ):
                # If reflection is missing, generate it

                reflection = reflection = (
                    reflections[i]
                    if i < len(reflections)
                    and reflections[i].step_number == step_obj.step_number
                    else self.judge_step(step_obj, task)
                )
                if reflection is not None and isinstance(reflection, Reflection):
                    reflection.step_number = step_obj.step_number
                    step_obj.reflection = reflection
                    interaction.reflections.append(reflection)
                else:
                    # Assign the reflection and reward to the step
                    step_obj.reflection = self.judge_step(step_obj, task)
            assert isinstance(
                step_obj.reflection, Reflection
            ), "Reflection object is not properly instantiated."
            assert hasattr(
                step_obj.reflection, "content"
            ), "Reflection object does not have a content attribute."
            print_saver.print_and_store(
                f"Reflection for step {step_obj.step_number}: {step_obj.reflection}"
            )
            print_saver.print_and_store(
                f"Type of reflection: {type(step_obj.reflection)}"
            )
            repair_log.append(
                f"Step {step_obj.step_number} reflection is missing. Generated reflection: {step_obj.reflection}"
            )
            if (
                interaction.reflections
                and len(interaction.reflections) > 0
                and interaction.reflections != []
            ):
                print_saver.print_and_store(
                    f"Type of reflection in interaction: {type(interaction.reflections)} and type of first item: {type(interaction.reflections[0])}"
                )
            else:
                print_saver.print_and_store(
                    f"Type of reflection in interaction: {type(interaction.reflections)} and type of reflection in step: {type(step_obj.reflection)}"
                )
                assert isinstance(
                    step_obj.reflection, Reflection
                ), "Reflection object is not properly instantiated."

            if step_obj.reflection not in interaction.reflections:
                interaction.reflections.append(step_obj.reflection)

            # Handle rewards if not already set
            if (
                step_obj.reflection
                and step_obj.reflection.reward == 0.0
                and i < len(rewards)
            ):
                step_obj.reflection.reward = self.judge_step(step_obj, task)

        # ===========================
        # Step 6: Extract Answer from Response
        # ===========================

        answer_match = re.search(
            r"<answer>(.*?)(?:</answer>|<final_reward>)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if answer_match:
            interaction.answer = answer_match.group(1).strip()

        # ===========================
        # Step 7: Extract Final Reward from Response
        # ===========================

        final_reward_match = re.search(
            r"<final_reward>(0\.\d+?|1\.0)</final_reward>",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if final_reward_match:
            interaction.final_reward = float(final_reward_match.group(1))

        # ===========================
        # Final Assertion
        # ===========================

        assert isinstance(
            interaction, Interaction
        ), "Interaction object is not properly instantiated."

        return interaction

    # ===========================
    # Example Usage
    # ===========================
    def parse_response(
        self,
        response: str,
        task: Task,
        steps_objs=None,
        reflections_objs=None,
        current_step_number: int = 0,
        current_remaining_budget: int = 0,
        interaction: Interaction = None,
        initial_budget: int = 0,
        plan_step_number: int = 0,
    ) -> Interaction:
        """
        Parses the OpenAI API response to extract steps, reflections, answers, and rewards.
        """
        # TODO: Implement a more robust response parsing mechanism, like using structured output from the model
        repair_log = []
        if interaction is None:
            interaction = Interaction(
                task=task, steps=[], reflections=[], answer="", final_reward=0.0
            )

        if response is None or not isinstance(response, str):
            return interaction

        # Check for any missing tags
        if not re.search(r"<step>", response, re.DOTALL | re.IGNORECASE):
            print_saver.print_and_store("No steps found in response.")
            return interaction
        if not re.search(r"<thinking>", response, re.DOTALL | re.IGNORECASE):
            print_saver.print_and_store("No <thinking> tags found in response.")
            return interaction
        thoughts = re.findall(
            r"<thinking>(.*?)<(?:/thinking|step|reflection|count|reward)>",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        # Extract steps
        if steps_objs is None or not isinstance(steps_objs, list):
            steps_objs = []

        if reflections_objs is None or not isinstance(reflections_objs, list):
            reflections_objs = []

        steps = re.findall(
            r"<step>(.*?)<(?:\/step|reflection|reward|step)>",
            response,
            re.DOTALL | re.IGNORECASE,
        )

        print_saver.print_and_store(f"Steps: {steps}")
        if not re.search(r"<count>", response, re.DOTALL | re.IGNORECASE):
            print_saver.print_and_store("No <count> tags found in response.")
            # Instead, we can use the current_remaining_budget, current_step_number, initial_budget, and steps_objs to infer the count values
            # Use current_remaining_budget, current_step_number, initial_budget, and steps_objs to infer the count values
            counts = []
            if steps_objs and len(steps_objs) > 0:
                # Get max remaining budget from steps_objs
                max_budget = max([s.remaining_budget for s in steps_objs]) + 1
                first_count = max(
                    max_budget, initial_budget if initial_budget > 0 else max_budget
                )

                # Generate counts backwards from first_count
                for i in range(len(steps)):
                    if (
                        i < len(steps_objs)
                        and steps_objs[i].remaining_budget is not None
                    ):
                        counts.append(steps_objs[i].remaining_budget + 1)
                    else:
                        prev_count = int(counts[-1]) if counts else first_count
                        counts.append(prev_count - 1)
            else:
                # If no steps_objs, generate counts based on initial_budget or current values
                first_count = (
                    initial_budget
                    if initial_budget > 0
                    else (current_remaining_budget + current_step_number)
                )
                counts = [str(first_count - i) for i in range(len(steps))]
                # Instead of <count> tags, infer counts based on existing parameters

            print_saver.print_and_store(f"Inferred counts: {counts}")
            repair_log.append(
                f"Could not find <count> tags in response. Inferred counts: {counts}"
            )
        else:
            counts = re.findall(
                r"<count>(.*?)<(?:\/count|thinking|step|reflection|reward|count)>",
                response,
                re.DOTALL | re.IGNORECASE,
            )
        first_count = re.search(
            r"<count>(.*?)<(?:\/count|thinking|step|reflection|reward|count)>",
            response,
            re.DOTALL | re.IGNORECASE,
        )  # Represents the initial step budget
        if first_count and first_count.group(1).strip().isnumeric():
            first_count = int(first_count.group(1))
        elif steps_objs is not None and steps_objs != []:

            first_count = max([s.remaining_budget for s in steps_objs]) + 1
            repair_log.append(
                f"first_count adjusted to {first_count} based on steps_objs."
            )
        elif initial_budget > len(steps) or initial_budget > len(steps_objs):
            first_count = initial_budget
            repair_log.append(
                f"first_count adjusted to {first_count} based on initial_budget."
            )
        if initial_budget != 0 and first_count != 0:
            if first_count != initial_budget:
                print_saver.print_and_store(
                    f"Initial budget mismatch. Adjusting initial budget. {first_count} vs initial: {initial_budget}"
                )
                repair_log.append(
                    f"Initial budget mismatch. Adjusting initial budget. {first_count} vs initial: {initial_budget}. Adjusting to {first_count}."
                )
                initial_budget = first_count

        elif initial_budget != 0 and isinstance(first_count, int) and first_count == 0:
            first_count = initial_budget
            print_saver.print_and_store(
                f"Initial budget mismatch. Adjusting initial budget. {first_count}"
            )
        elif initial_budget == 0 and isinstance(first_count, int) and first_count == 0:
            initial_budget = 12
            first_count = initial_budget
        elif (
            initial_budget == 0
            and isinstance(first_count, int)
            and first_count != 0
            and first_count is not None
        ):
            initial_budget = first_count
        else:
            initial_budget = 12
            first_count = initial_budget

        if steps_objs and max([s.remaining_budget for s in steps_objs]) + 1 > int(
            first_count
        ):
            first_count = int(max([s.remaining_budget for s in steps_objs]) + 1)
            print_saver.print_and_store(
                f"first_count adjusted to {first_count} based on steps_objs."
            )

        # Extract reflections
        # Revert reflections to the original pattern
        reflections = re.findall(
            r"<reflection>(.*?)<(?:\/reflection|thinking|step|count|reward|reflection)>",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        # Use the modified pattern for rewards
        rewards = re.findall(
            r"</reflection>\s*.*?<reward>(0\.\d+?|1\.0)<(?:/reward|thinking|step|reflection|count|reward?)>",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        # ensure each entry in rewards is a string before processing
        for i in range(len(rewards)):
            if isinstance(rewards[i], re.Match):
                rewards[i] = str(rewards[i].group(1))

        print_saver.print_and_store(f"Rewards: {rewards}")
        for iii in range(len(rewards)):
            print_saver.print_and_store(f"Step {iii + 1} reward: {rewards[iii]}")
        reflections = [
            reflection for reflection in reflections if reflection.strip() != ""
        ]
        if reflections_objs is not None and reflections_objs != []:
            reflections_objs = [
                r for r in reflections_objs if r.content.strip() != "" and r.content
            ]
            for sobj in steps_objs:
                if isinstance(sobj.reflection, Reflection):
                    reflections_objs.append(sobj.reflection)
            reflections_objs.sort(key=lambda r: r.step_number)
        else:
            reflections_objs = [
                sobj.reflection
                for sobj in steps_objs
                if isinstance(sobj.reflection, Reflection)
            ]
            reflections_objs.sort(key=lambda r: r.step_number)

        counts = [c.strip() for c in counts if c.isnumeric()]

        if (
            current_step_number is not None
            and current_step_number != 0
            and (len(steps) == 1 or len(steps_objs) == 1)
            and not (len(steps) == 1 and len(steps_objs) == 1)
        ):
            # This means that we are parsing a single step and the current_step_number is not 0, which means that the current step is not the first step. Simply return the interaction object fully populated with the current step and reflection, etc.
            if len(steps) == 1:
                if steps_objs is not None and steps_objs != []:
                    steps_objs[0].remaining_budget = (
                        current_remaining_budget
                        if current_remaining_budget != 0
                        else self.config.initial_budget - current_step_number + 1
                    )
                    steps_objs[0].step_number = current_step_number
                    steps_objs[0].reflection = (
                        reflections_objs[0]
                        if reflections_objs
                        else self.judge_step(steps_objs[0], task)
                    )
                    steps_objs[0].reflection.step_number = current_step_number
                    interaction.steps = steps_objs
                    interaction.reflections = reflections_objs
                    return interaction
                else:
                    steps_objs = [
                        Step(
                            steps[0],
                            current_step_number,
                            (
                                current_remaining_budget
                                if current_remaining_budget != 0
                                else self.config.initial_budget
                                - current_step_number
                                + 1
                            ),
                            (
                                reflections_objs[0]
                                if reflections_objs
                                else self.judge_step(steps[0], task)
                            ),
                            thoughts[0] if thoughts else None,
                            plan_step_number,
                        )
                    ]
                    interaction.steps = steps_objs
                    interaction.reflections = reflections_objs
                    return interaction

        # Combine existing and new reflections, avoiding duplicates
        existing_contents = {r.content.strip() for r in reflections_objs}
        new_reflections = [
            Reflection(
                content=r,
                step_number=(
                    first_count - int(counts[ii]) + 1
                    if ii < len(counts)
                    else (
                        reflections_objs[-1].step_number + 1 if reflections_objs else 1
                    )
                ),
                reward=(
                    float(rewards[ii])
                    if ii < len(rewards)
                    else self.judge_step(
                        Step(
                            steps[ii],
                            (
                                first_count - int(counts[ii]) + 1
                                if ii < len(counts)
                                else ii + 1
                            ),
                            initial_budget - (ii + 1),
                        ),
                        task,
                    ).reward
                ),
            )
            for ii, r in enumerate(reflections)
            if r not in existing_contents
        ]
        for refobj in reflections_objs:
            if (
                refobj not in new_reflections
                and refobj.content not in existing_contents
            ):
                if refobj.step_number is None:
                    refobj.step_number = (
                        first_count - int(counts[reflections_objs.index(refobj)]) + 1
                        if reflections_objs.index(refobj) < len(counts)
                        else (
                            reflections_objs[-1].step_number + 1
                            if reflections_objs
                            else 1
                        )
                    )
                elif refobj.step_number in [
                    r.step_number for r in new_reflections + reflections
                ]:
                    for r in new_reflections + reflections:
                        if r.step_number == refobj.step_number:
                            r.step_number += 1
                new_reflections.append(refobj)
        all_reflections = reflections_objs + new_reflections
        # Assign sequential step numbers
        all_reflections.sort(key=lambda r: r.step_number)
        for idx, reflection in enumerate(all_reflections, start=1):
            reflection.step_number = idx
        final_reflections = all_reflections

        reflections = final_reflections

        # check if both steps_objs and steps are empty
        if steps_objs is None and steps is None:
            # Check if the other parameters are empty
            if reflections is None and rewards is None and counts is None:
                return interaction
            else:
                # Check the <thinking> tags for more information
                if re.search(r"<thinking>", response):
                    print_saver.print_and_store(
                        "No steps found in response. Checking for <thinking> tags."
                    )
                    thoughts = re.findall(
                        r"<thinking>(.*?)<(?:/thinking|step|reflection|count|reward)>",
                        response,
                        re.DOTALL | re.IGNORECASE,
                    )
                    for thought in thoughts:
                        print_saver.print_and_store(f"Thought: {thought}")
                    return interaction
                else:
                    print_saver.print_and_store(
                        "No steps found in response. No <thinking> tags found."
                    )
                    return interaction
        else:
            # If any steps are empty but are present in steps_objs, use the descriptions from steps_objs. If any are empty but are in the right place and accompanied by a count, thinking, reflection, or reward tag, they will be filled in later.
            # Initialize first_count if not already set
            def calculate_expected_counts(first_count, index):
                """Calculate expected counts up to the given index."""
                if index == 0:
                    return [first_count]
                return list(range(first_count, first_count - (index + 1), -1))

            def validate_counts(expected, actual):
                """Validate if actual counts match expected counts."""
                return actual == expected

            def replace_empty_step(steps, steps_objs, index):
                """Replace an empty step with the description from steps_objs."""
                steps[index] = steps_objs[index].description
                print_saver.print_and_store(
                    f"Replaced empty step at index {index} with: '{steps_objs[index].description}'."
                )

            # Initialize first_count if not already set
            if (
                first_count is None
                or first_count == 0
                or not isinstance(first_count, int)
            ):
                if counts:
                    try:
                        first_count = int(counts[0].strip())
                    except ValueError:
                        print_saver.print_and_store(
                            f"Invalid integer in counts at index 0. Setting first_count to initial_budget {initial_budget}."
                        )
                        first_count = initial_budget
                else:
                    first_count = initial_budget

            for i, step in enumerate(steps):
                if step.strip() == "" and i < len(steps_objs):
                    if i < len(counts) and counts[i].strip() != "":
                        try:
                            # Calculate expected counts correctly
                            expected_counts = calculate_expected_counts(first_count, i)

                            # Extract actual counts up to the current index
                            actual_counts = [int(c.strip()) for c in counts[: i + 1]]

                            # Verify counts consistency
                            if validate_counts(expected_counts, actual_counts):
                                replace_empty_step(steps, steps_objs, i)
                            else:
                                # Find out how the counts are inconsistent. For example, if only the count that would have been at the current index is missing, we can replace the empty step with the description from steps_objs. Likewise, if the counts are all over the place, we can't make any assumptions.
                                if (
                                    len(actual_counts) < len(expected_counts)
                                    and len(expected_counts) - len(actual_counts) == 1
                                ):
                                    replace_empty_step(steps, steps_objs, i)
                                elif (
                                    len(actual_counts) < len(expected_counts)
                                    and len(expected_counts) - len(actual_counts) > 1
                                ):
                                    # For this missing count, check 1. whether the count is duplicated, 2. whether the duplicated count is the same as the one that would be at the current index, and 3. whether the thinking or reflection tags that would have been associated with it are also duplicated. If all conditions are met, we can remove the extra count and the corresponding thinking or reflection tag. If the current count is accurate after the removal, we can replace the empty step with the description from steps_objs.
                                    if (
                                        counts.count(counts[i]) > 1
                                        and counts[i] == expected_counts[i]
                                        and (
                                            reflections.count(reflections[i]) > 1
                                            or rewards.count(rewards[i]) > 1
                                        )
                                    ):
                                        counts.pop(i)
                                        if reflections.count(reflections[i]) > 1:
                                            reflections.pop(i)
                                        if rewards.count(rewards[i]) > 1:
                                            rewards.pop(i)
                                        if validate_counts(
                                            expected_counts, actual_counts
                                        ):
                                            replace_empty_step(steps, steps_objs, i)
                                elif len(actual_counts) > len(expected_counts):
                                    # Find the extra count by comparing the expected and actual counts as well as comparing the lengths of count and steps as well as thinking, reflection, and reward tags. Basically, if the counts are more than the steps, thinking, reflection, and reward tags, we can assume that the count is extra and can be removed. Likewise, if the duplicated count is the same as the one that would be at the current index AND either thinking or reflection tags are also duplicated, we can remove the extra count and the corresponding thinking or reflection tag. Either way, if the current count is accurate after the removal, we can replace the empty step with the description from steps_objs.
                                    if (
                                        len(actual_counts) > len(expected_counts)
                                        and len(counts) > len(steps)
                                        and len(counts) > len(reflections)
                                        and len(counts) > len(rewards)
                                    ):
                                        if counts.count(counts[i]) > 1 and (
                                            reflections.count(reflections[i]) > 1
                                            or rewards.count(rewards[i]) > 1
                                        ):
                                            counts.pop(i)
                                            if reflections.count(reflections[i]) > 1:
                                                reflections.pop(i)
                                            if rewards.count(rewards[i]) > 1:
                                                rewards.pop(i)
                                            if validate_counts(
                                                expected_counts, actual_counts
                                            ):
                                                replace_empty_step(steps, steps_objs, i)
                                        else:
                                            print_saver.print_and_store(
                                                f"Extra count found at index {i}. Unable to replace empty step."
                                            )
                                    elif (
                                        len(actual_counts) > len(expected_counts)
                                        and len(counts) == len(steps)
                                        and len(counts) == len(reflections)
                                        and len(counts) == len(rewards)
                                    ):
                                        # This condition being true means that the actual counts are more than the expected counts, but the counts, steps, reflections, and rewards are all the same length. This means that the count is extra and can be removed. If the duplicated count is the same as the one that would be at the current index AND either thinking or reflection tags are also duplicated, we can remove the extra count and the corresponding thinking or reflection tag. Either way, if the current count is accurate after the removal, we can replace the empty step with the description from steps_objs.
                                        if counts.count(counts[i]) > 1 and (
                                            reflections.count(reflections[i]) > 1
                                            or rewards.count(rewards[i]) > 1
                                        ):
                                            counts.pop(i)
                                            if reflections.count(reflections[i]) > 1:
                                                reflections.pop(i)
                                            if rewards.count(rewards[i]) > 1:
                                                rewards.pop(i)
                                            if validate_counts(
                                                expected_counts, actual_counts
                                            ):
                                                replace_empty_step(steps, steps_objs, i)
                                        elif (
                                            counts.count(counts[i]) > 1
                                            and counts[i] == expected_counts[i]
                                        ):
                                            counts.pop(i)
                                            if reflections.count(reflections[i]) > 1:
                                                reflections.pop(i)
                                            if rewards.count(rewards[i]) > 1:
                                                rewards.pop(i)
                                            if validate_counts(
                                                expected_counts, actual_counts
                                            ):
                                                replace_empty_step(steps, steps_objs, i)
                                        else:
                                            print_saver.print_and_store(
                                                f"Extra count found at index {i}. Unable to replace empty step."
                                            )
                        except ValueError:
                            print_saver.print_and_store(
                                f"Non-integer value found in counts at index {i}. Unable to replace empty step."
                            )
                    else:
                        print_saver.print_and_store(
                            f"Missing or empty count for step at index {i}. Unable to replace empty step."
                        )

            # Final validation
            for i, step in enumerate(steps):
                if step.strip() == "":
                    print_saver.print_and_store(
                        f"Warning: Step at index {i} remains empty after processing."
                    )

            # Additional validation for counts consistency
            try:
                if len(counts) >= len(steps):
                    expected_final_counts = list(
                        range(first_count, first_count - len(steps), -1)
                    )
                    actual_final_counts = [int(c.strip()) for c in counts[: len(steps)]]
                    if actual_final_counts != expected_final_counts:
                        print_saver.print_and_store(
                            f"Final counts mismatch. Expected {expected_final_counts}, got {actual_final_counts}."
                        )
            except ValueError:
                print_saver.print_and_store(
                    "Non-integer value found in counts during final validation."
                )

        interaction = self.process_steps(
            steps_objs=steps_objs,
            steps=steps,
            counts=counts,
            reflections=reflections,
            rewards=rewards,
            response=response,
            first_count=first_count,
            print_saver=print_saver,
            interaction=interaction,
            task=task,
            current_plan_step=plan_step_number,
            repair_log=repair_log,
        )

        return interaction

    def choose_best_response(
        self,
        responses: List[str],
        plan_step: PlanStep,
        interaction: Interaction,
        step_number: int,
    ) -> str:
        """
        Calls the OpenAI API with multiple responses and selects the best response based on the task and the responses consistency with the previous step, the task, and the overall interaction.
        """
        if not responses:
            return ""
        # Initialize variables
        best_response = ""
        resp_str = ""
        example_task = PlanStep(
            step_number=2,
            completed=False,
            step_name="Feature Engineering",
            step_description="Create new features",
            step_explanation="Feature engineering is the process of creating new features from existing data to improve the performance of machine learning models. This step involves transforming the data to make it more suitable for the model and extracting useful information from the data. Feature engineering can include creating new features, combining existing features, and transforming features to make them more informative.",
            step_output="New features created from the existing data",
            step_full_text="Create new features from existing data to improve the performance of machine learning models. This step involves transforming the data to make it more suitable for the model and extracting useful information from the data. Feature engineering can include creating new features, combining existing features, and transforming features to make them more informative.",
            subtasks=[],
        )
        previous_steps = [
            stp
            for stp in interaction.steps
            if stp.plan_step_number == plan_step.step_number
        ]
        previous_steps = [
            previous_step
            for previous_step in previous_steps
            if previous_step.step_number < step_number
        ]

        prev_step_strs = [
            f"<thinking>{stp_.thoughts}</thinking>\n<step>{stp_.description}</step>\n<reflection>{stp_.reflection}</reflection>\n"
            for stp_ in previous_steps
        ]

        token_cutoff = 4096
        prev_step_str = "".join(prev_step_strs)
        if len(prev_step_str) > token_cutoff:
            # Find the next tag after the cutoff
            match = re.search(
                r"<[^>]+>", prev_step_str[token_cutoff:], re.DOTALL | re.IGNORECASE
            )
            if match:
                next_tag_end = token_cutoff + match.end()
                prev_step_str = prev_step_str[:next_tag_end]
            else:
                # Fall back to direct cutoff if no tag found
                prev_step_str = prev_step_str[:token_cutoff]

        planstep_string = f"{plan_step.step_name}: {plan_step.step_description}. {plan_step.step_explanation}. Expected output of step: {plan_step.step_output}. Full text of plan step: {plan_step.step_full_text}"
        example_task_string = f"{example_task.step_name}: {example_task.step_description}. {example_task.step_explanation}. Expected output of step: {example_task.step_output}. Full text of example task: {example_task.step_full_text}"
        for i, response in enumerate(responses):
            resp_str += f"Response {i + 1}: {response}\n"
        messages = [
            {
                "role": "system",
                "content": "You are an expert in selecting the best response from a list of responses. You base your decision on the task, the consistency of the response with the previous step, and the overall interaction. Please select the best response from the list below:",
            },
            {
                "role": "user",
                "content": "Example Task: "
                + f"{example_task_string}"
                + "\n"
                + "Example Steps: "
                + "<step>#Input Data: Load the dataset into a pandas DataFrame \n import pandas as pd \n df = pd.read_csv('data.csv')</step> \n <step>#Process Data: Check for missing values \n missing_values = df.isnull().sum()</step> \n <step>#Feature Engineering: Create new features \n df['new_feature'] = df['feature1'] + df['feature2']</step> \n"
                + "Please select the response from the list of responses below that best represents the next step: \n"
                + "Responses: "
                + "Response 1: '<step>#Perform Exploratory Data Analysis (EDA): Analyze the dataset to understand the data distribution and relationships between features \n import matplotlib.pyplot as plt \n plt.hist(df['feature1'])</step>' \n"
                + "Response 2: '<step>#Visualize Data: Create a scatter plot to show the relationship between two features \n plt.scatter(df['feature1'], df['feature2'])</step>' \n"
                + "Response 3: '<step>#Feature Selection: Select the most important features for the model \n from sklearn.feature_selection import SelectKBest \n selector = SelectKBest(k=10)</step>' \n"
                + "\n"
                + "Respond with the number of the best response from the list above only.",
            },
            {
                "role": "assistant",
                "content": "1",
            },
            {
                "role": "system",
                "content": "Task: "
                + f"{planstep_string}"
                + "\n"
                + "Previous Steps: "
                + f"{prev_step_str}"
                + "\n"
                + "Please select the response from the list of responses below that best represents the next step: \n"
                + "Responses: "
                + resp_str
                + "\n"
                + "Respond with the number of the best response from the list above only.",
            },
        ]
        decision_response = None
        decision = None
        base_delay = 1
        max_delay = 16
        for attempt in range(self.config.max_retries):
            try:
                decision_response = openai.beta.chat.completions.parse(
                    model=self.config.model,
                    messages=messages,
                    response_format=response_choices,
                )
                decision = decision_response.choices[
                    0
                ].message.parsed.number_of_selected_response
                print_saver.print_and_store(
                    f"Decision in choose_best_response: {decision}. Responses available: {responses}"
                )
                if decision:
                    best_response = responses[int(decision) - 1]
                return (
                    best_response if best_response != "" else random.choice(responses)
                )
            except Exception as e:
                if decision_response:
                    print_saver.print_and_store(
                        f"Error in choose_best_response: {e}. Using default decision.  Responses available: {responses}. \n Decision response: {decision_response}"
                    )
                else:
                    print_saver.print_and_store(
                        f"Error in choose_best_response: {e}. Using default decision.  Responses available: {responses}."
                    )

                wait = min(base_delay * (2**attempt), max_delay)
                time.sleep(wait + random.uniform(0, 1))  # Adding jitter

        if decision:
            best_response = responses[int(decision) - 1]
        else:
            decision = responses.index(max(responses, key=len))
            best_response = responses[decision] if decision >= 0 else ""
        return best_response if best_response != "" else random.choice(responses)

    # -------------------------------
    # Self-Consistency
    # -------------------------------
    def self_consistency(
        self,
        task: Task,
        prompt: str,
        existing_interaction: Interaction,
        messages: List[dict],
        output_type: str,
        step_budget: int,
        step_number: int,
        plan_step_number: int,
        restart_limit: int = 3,
        backtrack_limit: int = 3,
        consistency_multiplier: int = 3,
        max_plan_steps: int = 0,
    ) -> Interaction:
        """
        Implements Self-Consistency by generating multiple reasoning paths and selecting the most consistent one.
        """
        responses = []
        steps_objs = existing_interaction.steps if existing_interaction else None
        reflections_objs = (
            existing_interaction.reflections if existing_interaction else None
        )
        msgs = messages.copy()
        for _ in range(self.config.n if self.config.n > 4 else 5):
            temperature = max(
                random.uniform(0.0, 0.5 + (min(0.1 * max(self.config.n, 5), 1.2))), 0.01
            )
            print_saver.print_and_store(f"Temperature: {temperature}")
            new_steps = []
            _thinking = []
            step_completion = False
            _reflections = []
            step_num = step_number
            restart_attempts = 0
            answer_response = None
            while len(new_steps) < step_budget and not step_completion:

                step_responses = []
                for _ in range(consistency_multiplier):
                    step_response = self.call_openai(
                        messages=msgs,
                        temperature=max(
                            random.uniform(
                                max(
                                    temperature
                                    - (min(temperature * max(self.config.n, 5), 1.2)),
                                    0.1,
                                ),
                                min(
                                    2.0,
                                    temperature
                                    + (min(temperature * max(self.config.n, 5), 1.2)),
                                ),
                            ),
                            0.01,
                        ),
                        stop_sequence=["</answer>"],
                    )
                    if step_response:
                        step_responses.append(step_response)
                plan_step_index = 0
                for i, stp in enumerate(task.plan.steps):
                    if stp.step_number == plan_step_number:
                        plan_step_index = i
                        break
                if step_responses == []:
                    print_saver.print_and_store(
                        f"No responses generated for step {step_num}."
                    )
                    break
                response = self.choose_best_response(
                    responses=step_responses,
                    plan_step=task.plan.steps[plan_step_index],
                    interaction=existing_interaction,
                    step_number=step_num,
                )

                if response:
                    msgs.append({"role": "system", "content": response})
                    print_saver.print_and_store(
                        f"Response in self-consistency: {response} for step {step_num}."
                    )
                    # Function to get the latest match using re.finditer

                def get_latest(pattern, text):
                    matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
                    latest = None
                    for match in matches:
                        latest = match.group(1)
                    return latest

                # Extract relevant tags from response with flexible stopping conditions
                latest_step = get_latest(
                    r"<step>(.*?)(?=<(?:\/?step|reflection|reward|thinking|count)>|$)",
                    response,
                )

                if latest_step is None or latest_step.strip() == "":
                    # look for answer tag instead
                    latest_step = get_latest(
                        r"<answer>(.*?)(?=<(?:\/?answer|reflection|reward|thinking|count)>|$)",
                        response,
                    )
                latest_thinking = get_latest(
                    r"<thinking>(.*?)(?=<(?:\/?thinking|step|reflection|reward|count)>|$)",
                    response,
                )
                latest_count = get_latest(
                    r"<count>(\d+)(?=<(?:\/?count|step|reflection|reward|thinking)>|$)",
                    response,
                )
                latest_reflection = get_latest(
                    r"<reflection>(.*?)(?=<(?:\/?reflection|step|thinking|reward|count)>|$)",
                    response,
                )
                latest_reward = get_latest(
                    r"<reward>(.*?)(?=<(?:\/?reward|step|reflection|thinking|count)>|$)",
                    response,
                )

                # Create a response interaction using just the latest chunks
                try:
                    current_step = Step(
                        description=latest_step,
                        step_number=step_num,
                        remaining_budget=(
                            int(latest_count)
                            if latest_count
                            else (
                                int(new_steps[-1].remaining_budget - 1)
                                if new_steps
                                else int(step_budget)
                            )
                        ),
                        plan_step_number=plan_step_number,
                    )
                except ValidationError as e:
                    print_saver.print_and_store(
                        f"Validation error in self-consistency: {e}. Response: {response}"
                    )
                    print_saver.print_and_store(
                        f"Messages were: \n"
                        + "\n".join([f"{m['role']}: {m['content']}" for m in msgs])
                    )
                    raise e
                # step_completion = self.judge_step_completion(

                if latest_reflection and latest_reward:
                    reflection = Reflection(
                        content=latest_reflection,
                        reward=float(latest_reward),
                        step_number=current_step.step_number,
                    )
                    current_step.reflection = reflection
                    _reflections.append(reflection)
                else:
                    reflection = self.judge_step(current_step, task)
                    current_step.reflection = reflection
                    _reflections.append(reflection)
                backtracks = 0
                while (
                    (
                        current_step.reflection.reward < 0.8
                        and current_step.reflection.reward > 0.5
                    )
                    and backtracks < backtrack_limit
                ) or (
                    (
                        current_step.reflection.reward < 0.5
                        and backtracks < backtrack_limit
                    )
                    and restart_attempts < restart_limit
                ):
                    former_score = current_step.reflection.reward
                    msgs_b = msgs.copy()
                    msgs_b.append(
                        {
                            "role": "user",
                            "content": f"You have been rated {current_step.reflection.reward} for your reasoning. The review is as follows: {current_step.reflection.content}. Please rewrite the last step based on the feedback.",
                        }
                    )
                    revision = self.call_openai(
                        messages=msgs_b,
                        temperature=temperature + max(random.uniform(-0.1, 0.1), 0.01),
                        stop_sequence=["</step>"],
                    )
                    if revision:
                        msgs.remove(msgs[-1])
                        msgs.append({"role": "system", "content": revision})
                        print_saver.print_and_store(
                            f"Revision in self-consistency: {revision} for step {step_num}."
                        )

                        # remove count from revision
                        latest_count = get_latest(
                            r"<count>(.*?)(?=<(?:\/?count|step|reflection|reward|thinking)>|$)",
                            revision,
                        )
                        if latest_count:
                            revision = revision.replace(
                                f"<count>{latest_count}</count>", ""
                            )
                        revision += "</step>"
                        latest_step = get_latest(
                            r"<step>(.*?)(?=<(?:\/?step|reflection|reward|thinking|count)>|$)",
                            response,
                        )
                        current_step.description = latest_step
                        current_step.reflection = self.judge_step(current_step, task)
                        if _reflections:
                            _reflections[-1] = current_step.reflection
                        backtracks += 1
                if latest_thinking:
                    _thinking.append(latest_thinking)
                    current_step.thoughts = latest_thinking  # Add thoughts to the step
                new_steps.append(current_step)
                if (
                    re.search(r"<answer>", response, re.DOTALL | re.IGNORECASE)
                    is not None
                ):
                    response += "</answer>"
                answer_response = get_latest(r"<answer>(.*?)</answer>", response)
                if answer_response is not None or len(new_steps) >= step_budget:
                    reason = (
                        f"Got answer: {answer_response}."
                        if answer_response
                        else f"Length of steps ({len(new_steps)})reached the budget, {step_budget}."
                    )
                    print_saver.print_and_store(
                        f"Steps {step_number} to {step_num} completed. Reasoning path: {new_steps}. Reason: {reason}"
                    )
                    plan_step_index = 0
                    for i, stp in enumerate(task.plan.steps):
                        if stp.step_number == plan_step_number:
                            plan_step_index = i
                            break
                    if self.judge_step_completion(
                        new_steps, task.plan.steps[plan_step_index], max_plan_steps
                    )[0]:
                        print_saver.print_and_store(f"Was judged as complete.")
                        responses.append(
                            Interaction(
                                task=task,
                                steps=new_steps,
                                reflections=_reflections,
                                answer=answer_response if answer_response else "",
                            )
                        )
                        if answer_response is not None:
                            responses[-1].answer = answer_response
                            responses[-1].final_reward = (
                                float(
                                    get_latest(
                                        r"<final_reward>(.*?)</final_reward>", response
                                    )
                                )
                                if get_latest(
                                    r"<final_reward>(.*?)</final_reward>", response
                                )
                                else self.judge_final_answer(
                                    task, responses[-1]
                                ).final_reward
                            )
                            print_saver.print_and_store(
                                f"Final answer for reasoning path: {responses[-1].answer} with reward: {responses[-1].final_reward}."
                            )
                        step_completion = True
                        break

                    elif (
                        restart_attempts < restart_limit
                        and current_step.reflection.reward < 0.5
                        and former_score < 0.5
                    ):

                        restart_attempts += 1
                        msgs = messages.copy()
                        new_steps = []
                        _thinking = []
                        _reflections = []
                        step_num = step_number
                        step_completion = False
                        answer_response = None
                        print_saver.print_and_store(
                            f"Was judged as incomplete. Restarting reasoning path. Attempt {restart_attempts} of {restart_limit}."
                        )
                    else:
                        responses.append(
                            Interaction(
                                task=task,
                                steps=new_steps,
                                reflections=_reflections,
                                answer=answer_response if answer_response else "",
                            )
                        )
                        break
                step_num += 1

        # Aggregate responses (select the answer with the highest final reward)
        best_interaction: Interaction = max(
            responses, key=lambda x: x.final_reward if x.final_reward else 0.0
        )
        assert isinstance(best_interaction, Interaction)
        # Add the best interaction to the existing interaction
        if existing_interaction:
            existing_interaction.steps = (
                best_interaction.steps
                if len(best_interaction.steps) > len(existing_interaction.steps)
                else existing_interaction.steps
            )
            existing_interaction.reflections = (
                best_interaction.reflections
                if len(best_interaction.reflections)
                > len(existing_interaction.reflections)
                else existing_interaction.reflections
            )
            existing_interaction.answer = best_interaction.answer
            existing_interaction.final_reward = (
                best_interaction.final_reward
                if best_interaction.final_reward and best_interaction.final_reward > 0.0
                else 0.0
            )
            return existing_interaction
        return best_interaction

    # -------------------------------
    # Automatic Prompt Engineering (APE)
    # -------------------------------
    def automatic_prompt_engineering(self, task: str) -> str:
        """
        Automatically refines the prompt based on previous interactions.
        """
        # TODO - Implement more advanced APE strategies
        prompt = f"Please solve the following task using advanced reasoning techniques: '{task}'"
        return prompt

    # -------------------------------
    # Final Prompt Generation
    # -------------------------------
    def generate_final_prompt(self, task: str) -> str:
        """
        Generates the final prompt by incorporating retrieved information and collaborative agent responses.
        """
        # Retrieve additional information if available
        retrieved_info = self.retrieve_information(task)

        # Generate prompts from multiple agents
        agent_prompts = self.collaborative_reasoning(task)

        # Compile the final prompt
        final_prompt = f"{retrieved_info}\n"
        final_prompt += self.tag_based_prompt(task)
        final_prompt += "\n<agents_responses>\n"
        for idx, agent_response in enumerate(agent_prompts):
            final_prompt += f"Agent {idx+1} Response:\n{agent_response}\n"
        final_prompt += "</agents_responses>"

        return final_prompt

    def apply_prompt_modifications(self, modifications: PromptSuggestions) -> str:
        """
        Applies the prompt modifications suggested by the AI using another LLM call.
        """
        # Generate a new prompt based on the suggested modifications
        new_prompt = (
            f"Modify the prompt for the task based on the following suggestions:\n\n"
        )
        for idx, suggestion in enumerate(modifications.prompt_modifications, start=1):
            new_prompt += f"{idx}. {suggestion}\n"

        messages = [
            {
                "role": "system",
                "content": "You are an expert prompt engineer. You understand many aspects of prompt engineering and are knowledgeable about many SOTA prompting techniques. You use this knowledge to analyze and improve prompts.",
            },
            {"role": "user", "content": new_prompt},
        ]

        # Call the OpenAI API to generate the modified prompt
        response = openai.chat.completions.create(
            model=self.config.model,
            messages=messages,
        )

        return response.choices[0].message.content.strip()

    def task_into_prompt(self, task: str) -> str:
        """
        Converts the task description into a concise prompt for the AI, using AI.
        """

        # Generate the initial prompt for the task
        first_prompt = self.automatic_prompt_engineering(task)

        # Call the OpenAI API to generate the refined prompt
        response = openai.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an AI assistant tasked with solving complex problems. Your job is to provide a clear and concise prompt to guide the reasoning process for the given task.
                The prompt should be concise while providing all necessary information to solve the task effectively. Ensure that the prompt is simple but detailed, focusing only on the given task without straying into irrelevant details or steps beyond the scope of this task.
                
                Please output the refined prompt enclosed within <prompt> tags.
                Also, word the prompt in a way that encourages critical thinking and systematic problem-solving.
                Finally, word the prompt using active voice, using the same verbs as the task description and directly addressing the LLM receiving the prompt with direct instructions that will ensure it understands the task and can provide a solution effectively.

                Example:
                Task: 'Write a short story about a robot learning to understand human emotions.'
                <prompt>Write a short story where a robot gradually learns to understand human emotions through interactions with a diverse group of people. Focus on the robot's internal thoughts and the challenges it faces in interpreting emotions.</prompt>
                
                Another Example:
                Task: 'Calculate the derivative of f(x) = sin(x) * e^x.'
                <prompt>Calculate the derivative of the function f(x) = sin(x) * e^x using the product rule of differentiation. Show all steps clearly and explicitly, ensuring to simplify the final expression.</prompt

                Third Example:
                Task: 'Analyze the impact of climate change on global food security.'
                <prompt>Analyze the impact of climate change on global food security by examining the effects on crop yields, food production, and distribution systems. Consider both short-term and long-term consequences, and propose potential solutions to mitigate these impacts.</prompt>
                
                """,
                },
                {
                    "role": "user",
                    "content": f"Refine the prompt for the following task: '{task}'",
                },
            ],
            max_completion_tokens=200,
        )

        # Extract the refined prompt from the response
        pattern = re.compile(r"<prompt>(.*?)</prompt>", re.DOTALL | re.IGNORECASE)
        match = pattern.search(response.choices[0].message.content)
        refined_prompt = match.group(1) if match else ""

        return refined_prompt if refined_prompt not in ["", " "] else first_prompt

        # -------------------------------
        # Main Workflow Integration
        # -------------------------------
        # def generate_step_prompt(
        #     self, full_prompt: str, interaction: Interaction, current_step: int, remaining_budget: int
        # ) -> str:
        #     """
        #     Generates the prompt for the current step by appending the interaction history.
        #     """
        #     latest_reflection = interaction.reflections[-1].content if interaction.reflections else ""
        #     latest_reward = interaction.reflections[-1].reward if interaction.reflections else 1.0

        #     if latest_reward < self.config.confidence_thresholds[1]:
        #         system_message = "You need to re-evaluate your approach based on the low reward score."
        #     else:
        #         system_message = "Continue with the current approach."

        #     step_prompt = f"{full_prompt}\n\n"
        #     step_prompt += f"Current Step: {current_step}\n"
        #     step_prompt += f"Latest Reflection: {latest_reflection}\n"
        #     step_prompt += f"Latest Reward: {latest_reward}\n"

        return step_prompt

    # def determine_expected_output(self, task: str) -> str:
    #     """
    #     Determines the expected output for the given task.
    #     """
    #     # Call the OpenAI API to generate the expected output
    #     response = openai.chat.completions.create(
    #         model=self.config.model,
    #         messages=[
    #             {
    #                 "role": "system",
    #                 "content": "You are an AI assistant tasked with determining the expected output for a given task. Your job is to analyze the task description and provide a clear and concise expected output that aligns with the task requirements.",
    #             },
    #             {"role": "user", "content": f"Given the task: '{task}', determine the expected output."},
    #         ],
    #     )

    #     return response.choices[0].message.content.strip()

    def merge_interactions(
        self, interaction_a: Interaction, interaction_b: Interaction
    ) -> Interaction:
        """
        Merges two interactions into a single interaction.
        """
        if not isinstance(interaction_a, Interaction) or not isinstance(
            interaction_b, Interaction
        ):
            raise TypeError("Both arguments must be Interaction instances")
        merged_interaction = Interaction(
            task=interaction_a.task,
            steps=[],
            reflections=[],
            answer="",
            final_reward=0.0,
        )
        try:

            # Merge steps
            if len(interaction_a.steps) < len(interaction_b.steps):
                interaction_a.steps.extend(
                    stp
                    for stp in interaction_b.steps
                    if stp not in interaction_a.steps
                    and stp.step_number
                    not in [s.step_number for s in interaction_a.steps]
                )
                merged_interaction.steps = interaction_a.steps
            elif len(interaction_a.steps) > len(interaction_b.steps):
                interaction_b.steps.extend(
                    stp
                    for stp in interaction_a.steps
                    if stp not in interaction_b.steps
                    and stp.step_number
                    not in [s.step_number for s in interaction_b.steps]
                )
                merged_interaction.steps = interaction_b.steps
            else:
                merged_interaction.steps = interaction_a.steps

            # Merge reflections
            merged_interaction.reflections = [
                ref
                for ref in [stepp.reflection for stepp in merged_interaction.steps]
                if ref is not None
            ]

            # Merge answer
            if interaction_b.answer is not None:
                interaction_a.answer = interaction_b.answer
            elif interaction_a.answer is not None:
                interaction_b.answer = interaction_a.answer

            # Merge final reward
            if interaction_b.final_reward is None:
                interaction_b.final_reward = 0.0

            merged_interaction.final_reward = (
                interaction_a.final_reward
                if interaction_a.final_reward and interaction_a.final_reward != 0.0
                else interaction_b.final_reward
            )

            return merged_interaction
        except Exception as e:
            print_saver.print_and_store(f"Error in merging interactions: {e}")
            raise RuntimeError(f"Error in merging interactions: {e}")

    def main(self, task: str) -> Interaction:
        """
        Integrates all components to process the task and generate the final answer.
        """
        # Step 1: Retrieve Information
        retrieved_info = self.retrieve_information(task)
        print_saver.print_and_store("Retrieved Information:\n" + retrieved_info)
        refined_task = self.task_into_prompt(task)
        print_saver.print_and_store("Refined Task:" + refined_task)

        _description = task
        _refined_description = refined_task
        _output_type = output_type_determination(task)

        # Step 2: Assess Complexity
        complexity, plan = self.assess_complexity(refined_task)
        task_object = Task(
            task, refined_task, complexity, [], [], "", 0.0, plan, _output_type
        )
        main_interaction = Interaction(
            task=task_object, steps=[], reflections=[], answer="", final_reward=0.0
        )
        print_saver.print_and_store("Assessed Complexity:" + str(complexity))
        assert complexity > 0.0, "Complexity assessment failed."
        assert isinstance(plan, Plan), "Complexity plan generation failed."
        task_object.project_name = self.name_project(task_object, plan)
        initial_step_budget = len(plan.steps)
        subtasks = []
        for pstep in plan.steps:
            subtasks.append(pstep.subtasks)

        # flatten the subtasks into a single list
        subtasks = [item for sublist in subtasks for item in sublist]
        initial_step_budget += len(subtasks)
        self.config.initial_budget = initial_step_budget

        task_object.complexity = complexity
        self.task_object = task_object
        self.config.complexity_factor = (
            len(subtasks) / len(plan.steps) if len(subtasks) > 0 else 5.0
        )

        # Step 3: Adjust Step Budget
        adjusted_budget = self.adjust_step_budget(refined_task, task_object.complexity)
        print_saver.print_and_store("Adjusted Step Budget:" + str(adjusted_budget))

        # Step 4: Generate Initial Prompt
        initial_prompt, system_prompt = self.generate_initial_prompt(
            refined_task,
            retrieved_info,
            adjusted_budget,
            complexity,
            task_object.output_type,
        )
        print_saver.print_and_store("System Prompt:\n" + system_prompt)
        print_saver.print_and_store("Initial Prompt:\n" + initial_prompt)
        current_planstep_prompt = initial_prompt

        # Find the step that has step_number = 1
        current_plan_step = next((x for x in plan.steps if x.step_number == 1), None)
        current_plan_subtask = None
        if current_plan_step is not None and current_plan_step.subtasks:
            current_plan_subtask = next(
                (
                    x
                    for x in current_plan_step.subtasks
                    if isinstance(x, Subtask) and x.subtask_number == 1
                ),
                None,
            )
        for step in range(adjusted_budget):

            assert current_plan_step is not None
            curr_num = None
            curr_name = None
            curr_desc = None
            curr_explanation = None
            curr_output = None
            curr_fulltext = None
            curr_st_num = None
            curr_st_name = None
            curr_st_desc = None
            curr_st_explanation = None
            curr_st_output = None
            curr_st_fulltext = None
            max_step = None
            if current_plan_step and isinstance(current_plan_step, PlanStep):
                curr_num = current_plan_step.step_number
                curr_name = current_plan_step.step_name
                curr_desc = current_plan_step.step_description
                curr_explanation = current_plan_step.step_explanation
                curr_output = current_plan_step.step_output
                curr_fulltext = current_plan_step.step_full_text
                max_step = len(plan.steps)
                print_saver.print_and_store(
                    f"Current Plan Step: {curr_num} - {curr_name}"
                )
                this_step_prompt = self.convert_planstep_to_prompt(
                    current_plan_step, task_object
                )
            else:
                raise ValueError("Invalid type for current_plan_step")
            if current_plan_subtask and isinstance(current_plan_subtask, Subtask):
                curr_st_num = current_plan_subtask.subtask_number
                curr_st_name = current_plan_subtask.subtask_name
                curr_st_desc = current_plan_subtask.subtask_description
                curr_st_explanation = current_plan_subtask.subtask_explanation
                curr_st_output = current_plan_subtask.subtask_output
                curr_st_fulltext = current_plan_subtask.subtask_full_text
                print_saver.print_and_store(
                    f"Current Plan Sub-Task: {curr_st_num} - {curr_st_name}"
                )

            condensed_plan = self.condense_plan(plan)
            step_prompt = f"""\n\nPlease focus on the completing following:
            - {curr_name}
            - Description: {curr_desc}
            - Basically, you need to: {curr_explanation}
            - The expected output for this step is: {curr_output}
            - Full Text: {curr_fulltext}

            """
            st_prompt = ""
            if curr_st_num is not None and current_plan_subtask is not None:
                st_prompt = f"""Specifically, for the current sub-task, focus on the following sub-task:
            - Sub-Task Name: {curr_st_name}
            - Sub-Task Description: {curr_st_desc}
            - Sub-Task Explanation: {curr_st_explanation}
            - Sub-Task Expected Output: {curr_st_output}
            - Sub-Task Full Text: {curr_st_fulltext}
            """

            step_prompt = this_step_prompt if this_step_prompt else step_prompt

            custom_init_prompt = initial_prompt + f"\n\n{step_prompt}"
            if st_prompt is not None and st_prompt != "":
                custom_init_prompt += f"\n{st_prompt}"
            count = adjusted_budget - step

            current_planstep_prompt.replace(initial_prompt, custom_init_prompt)
            steps_this_planstep = [
                stp
                for stp in main_interaction.steps
                if stp.plan_step_number == curr_num
            ]
            steps_this_planstep.sort(key=lambda x: x.step_number)
            assistant_tags = ""
            for stp in steps_this_planstep:
                if stp.remaining_budget == count:
                    assistant_tags += f"\n<count>{count}</count>\n"
                    assistant_tags += f"\n<thinking>"
                elif stp.remaining_budget < count:
                    count_ = stp.remaining_budget + 1
                    assistant_tags += f"<count>{count_}</count>"
                    thought_ = (
                        stp.thoughts.replace("<thinking>", "").replace(
                            "</thinking>", ""
                        )
                        if stp.thoughts
                        else ""
                    )
                    step = stp.description.replace("<step>", "").replace("</step>", "")
                    reflection_ = (
                        stp.reflection.content.replace("<reflection>", "").replace(
                            "</reflection>", ""
                        )
                        if stp.reflection
                        else ""
                    )

                    assistant_tags += f"<thinking>{thought_}</thinking>\n"
                    assistant_tags += f"<step>{step}</step>\n"
                    assistant_tags += f"<reflection>{reflection_}</reflection>\n"

            # Add a count tag to the prompt

            print_saver.print_and_store("Current Prompt:\n" + current_planstep_prompt)
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": current_planstep_prompt,
                    "name": "Human_User",
                },
                {
                    "role": "assistant",
                    "content": assistant_tags,
                },
            ]
            # Step 5: Collaborative Multi-Agent Reasoning
            # for collaborative_reasoning, input only the current prompt minus the initial prompt
            agent_prompt = current_planstep_prompt.replace(initial_prompt, "")
            if self.config.agents > 0:
                agent_interaction = self.collaborative_reasoning_main(
                    Task(
                        curr_desc,
                        self.task_into_prompt(curr_desc),
                        complexity,
                        [],
                        [],
                        "",
                        0.0,
                        plan,
                        _output_type,
                    ),
                    agent_prompt,
                    main_interaction,
                    task_object.output_type,
                    max(count / len([p for p in plan.steps if not p.completed]), 1),
                    step,
                    curr_num,
                    3,
                    max_step,
                )
            # Step 6: Self-Consistency Check
            self_consistent_interaction = self.self_consistency(
                Task(
                    curr_desc,
                    self.task_into_prompt(curr_desc),
                    complexity,
                    [],
                    [],
                    "",
                    0.0,
                    plan,
                    _output_type,
                ),
                current_planstep_prompt,
                main_interaction,
                messages,
                task_object.output_type,
                max(count / len([p for p in plan.steps if not p.completed]), 1),
                step,
                curr_num,
                3,
            )
            assert isinstance(self_consistent_interaction, Interaction)
            print_saver.print_and_store(
                "Self-Consistent Interaction:" + self_consistent_interaction.__str__()
            )

            for stp in self_consistent_interaction.steps:
                if stp.plan_step_number is None or stp.plan_step_number == 0:
                    stp.plan_step_number = curr_num
            for stp in agent_interaction.steps:
                if stp.plan_step_number is None or stp.plan_step_number == 0:
                    stp.plan_step_number = curr_num

            # Step 7: Aggregate and Select Best Interaction
            all_interactions = []
            all_interactions.append(agent_interaction)
            all_interactions.append(self_consistent_interaction)
            if any(
                [
                    ict.answer is not None and ict.answer.strip() != ""
                    for ict in all_interactions
                ]
            ):
                all_interactions = [
                    ict for ict in all_interactions if ict.answer is not None
                ]
            best_interaction = max(
                all_interactions,
                key=lambda x: x.sum_rewards() if x.sum_rewards() else 0.0,
            )

            assert (
                best_interaction.sum_rewards() is not None
                and best_interaction.sum_rewards() > 0.0
            ), f"Invalid reward value: {best_interaction.sum_rewards()}"

            # Step 8: Dynamic Confidence Exploration
            if best_interaction.final_reward < self.config.confidence_thresholds[1]:
                best_interaction = self.dynamic_confidence_exploration(
                    best_interaction, task_object, current_plan_step
                )

            # Step 9: Prompt Refinement
            if best_interaction.final_reward < self.config.confidence_thresholds[2]:
                refined_prompt = self.refine_prompt(best_interaction, task)
                best_interaction = self.self_consistency(task, refined_prompt)

            # Now, judge whether the latest plan step or sub-task was completed successfully
            assert isinstance(best_interaction, Interaction)

            assert isinstance(main_interaction, Interaction)
            planstep_index = 0
            for i, stp in enumerate(plan.steps):
                if stp.step_number == curr_num:
                    planstep_index = i
                    break
            # if best_interaction.steps and curr_num in [
            #     final_output.planstep.step_number
            #     for final_output in main_interaction.planstep_outputs
            # ]:
            #     for step in best_interaction.steps:
            #         if step not in main_interaction.steps:
            #             main_interaction.steps.append(step)
            #     for reflection in best_interaction.reflections:
            #         if reflection not in main_interaction.reflections:
            #             main_interaction.reflections.append(reflection)

            steps_this_planstep = [
                stp
                for stp in main_interaction.steps
                if stp.plan_step_number == curr_num
            ]
            if current_plan_step and isinstance(current_plan_step, PlanStep):
                if current_plan_subtask and isinstance(current_plan_subtask, Subtask):
                    complete, next_st = self.judge_subtask_completion(
                        best_interaction.steps[-1], current_plan_subtask
                    )
                    if complete:
                        current_plan_subtask.completed = True
                        num_completed = 0
                        for st in current_plan_step.subtasks:
                            if st.completed:
                                num_completed += 1
                            if (
                                st.subtask_number == next_st
                                and not st.completed
                                and st.subtask_number < len(current_plan_step.subtasks)
                            ):
                                current_plan_subtask = st
                                break
                        if num_completed == len(current_plan_step.subtasks):
                            current_plan_step.completed = True

                    else:
                        current_plan_subtask.completed = False
                complete, next_step_num = self.judge_step_completion(
                    steps_this_planstep, current_plan_step, max_step
                )
                if complete:
                    current_plan_step.completed = True
                    if (
                        best_interaction.steps
                        and task_object.plan.steps[planstep_index].completed
                    ):
                        for step in best_interaction.steps:
                            if step not in main_interaction.steps:
                                main_interaction.steps.append(step)
                        for reflection in best_interaction.reflections:
                            if reflection not in main_interaction.reflections:
                                main_interaction.reflections.append(reflection)

                    steps_this_planstep = [
                        stp
                        for stp in main_interaction.steps
                        if stp.plan_step_number == curr_num
                    ]
                    for stp in steps_this_planstep:
                        # get same step in main_interaction
                        for stp_ in main_interaction.steps:
                            previous_steps = [
                                previous_step
                                for previous_step in main_interaction.steps
                                if previous_step.plan_step_number
                                == stp.plan_step_number
                            ]
                            previous_steps = [
                                previous_step
                                for previous_step in previous_steps
                                if previous_step.step_number < stp.step_number
                            ]
                            if (
                                stp_.step_number == stp.step_number
                                and stp_.plan_step_number == stp.plan_step_number
                            ):
                                stp_.completed = True
                                step_index = main_interaction.steps.index(stp_)
                                final_step_output = self.finalize_step_output(
                                    stp_, task_object, current_plan_step, previous_steps
                                )
                                main_interaction.steps[step_index].final_step_output = (
                                    final_step_output
                                )
                                main_interaction.step_outputs.append(final_step_output)
                    final_planstep_output = self.finalize_planstep_output(
                        steps_this_planstep, task_object, current_plan_step
                    )
                    main_interaction.existing_files.append(
                        final_planstep_output.file_name
                    )
                    main_interaction.planstep_outputs.append(final_planstep_output)

                else:
                    current_plan_step.completed = False
                if (
                    current_plan_step.completed
                    and current_plan_step.step_number < len(plan.steps)
                    and current_plan_step.step_number
                    < max(plan.steps, key=lambda x: x.step_number).step_number
                ):

                    assert isinstance(main_interaction, Interaction)
                    plan.steps[current_plan_step.step_number].completed = True
                    next_step = None
                    for st in plan.steps:
                        if st.step_number == next_step_num:
                            next_step = st
                            break
                    current_plan_step = next_step
                    if current_plan_step and current_plan_step.subtasks:
                        current_plan_subtask = next(
                            (
                                x
                                for x in current_plan_step.subtasks
                                if isinstance(x, Subtask) and x.subtask_number == 1
                            ),
                            None,
                        )
                    else:
                        current_plan_subtask = None
            if main_interaction.answer and main_interaction.answer.strip() != "":
                print_saver.print_and_store(
                    f"\n\nFinal Answer: {main_interaction.answer}\n\n"
                )
                interaction = self.judge_final_answer(task, main_interaction)
                main_interaction.final_reward = interaction.final_reward
                return main_interaction
        else:
            return main_interaction

    def adaptive_complexity_handling(
        self, task: str, interaction: Interaction
    ) -> Interaction:
        """
        Adjusts the complexity and reprocesses the task if necessary.
        """
        complexity = self.assess_complexity(task)
        if complexity > 3:  # Threshold for high complexity
            adjusted_budget = self.adjust_step_budget(task, complexity)
            prompt = self.generate_initial_prompt(task)
            prompt = prompt.replace(
                f"Start with a {self.config.initial_budget}-step budget",
                f"Start with a {adjusted_budget}-step budget",
            )
            print_saver.print_and_store("Adjusting step budget due to high complexity.")
            response = self.call_openai(prompt)
            if response:
                new_interaction = self.parse_response(response)
                if new_interaction.final_reward and new_interaction.final_reward > (
                    interaction.final_reward or 0.0
                ):
                    return new_interaction
        return interaction


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    config = PromptEngineeringConfig()
    engineer = AdvancedPromptEngineer(config)
    task_description = """
Objective: Develop and implement a Python script for a custom strategy tile-based video game called "Resource Wars." The script must accomplish the following tasks:

Game Requirements:

1. Game Board and Rules:
   - Create a 10x10 grid-based game board.
   - Define three types of resources: Gold, Wood, and Stone.
   - Implement the following unit types:
     - Worker: Collects resources from adjacent tiles.
     - Warrior: Attacks opponent units.
     - Builder: Constructs buildings.
   - Define building types:
     - Base: Stores resources and spawns new units.
     - Barracks: Spawns Warriors.
     - Outpost: Expands player control.
   - Set the win conditions:
     - Accumulate a predefined amount of each resource.
     - Eliminate all opponent bases.

2. AI Player Implementation:
   - Create three levels of AI:
     - Beginner: Makes random but valid moves.
     - Intermediate: Uses heuristics to prioritize resource collection and basic defense.
     - Advanced: Implements strategic planning using minimax or Monte Carlo Tree Search (MCTS).

3. Turn-Based Logic:
   - Implement a turn-based system allowing each player to perform actions in sequence.
   - Validate and execute player actions: move units, collect resources, attack opponents, or build structures.

4. Dynamic Decision Making:
   - The AI players should adapt their strategies dynamically based on the current game state, including opponent moves.

5. Performance Evaluation:
   - Run simulated matches between AI players of different levels.
   - Log outcomes, decision-making processes, and overall performance for each AI.

Additional Notes:
- The game may be text-based or include a simple visualization (optional).
- Include error handling and validation to ensure robust gameplay mechanics.

Test Success Criteria:
- Functional game logic with turn-based mechanics.
- AI players exhibit behavior consistent with their defined strategies.
- Advanced AI demonstrates dynamic adaptation and strategic planning.
- All winning conditions and game rules are enforced correctly.
"""
    task_description = "Write a Python script to calculate the factorial of a given number using recursion."
    # TODO: Replace the task_description with steps from the plans generated by complexity_measures.py
    result = engineer.main(task_description)
    assert isinstance(
        result, Interaction
    ), f"Invalid result type: {type(result)} Results: {result}"
    result.save_final_outputs_to_logs()
    result.save_final_file()
    print_saver.print_and_store("\nFinal Interaction:")
    print_saver.print_and_store(result.__str__())

    print_saver.print_and_store("\nFinal Answer:")
    print_saver.print_and_store(result.answer)
    print_saver.print_and_store("\nFinal Reflection:")
    print_saver.print_and_store(result.reflections[-1].content)
    print_saver.print_and_store(f"Reward Score: {result.final_reward}")
    print_saver.save_prints()
