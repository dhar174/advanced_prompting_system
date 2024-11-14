from math import e
import openai
import os
import re
import time
import random

from regex import R
from tqdm import tqdm
from traitlets import Int
from conversation_manager import output_type_determination, OutputType
from pydantic import BaseModel, Field

from typing import List, Tuple, Optional


import complexity_measures
from complexity_measures import Plan, PlanStep, Subtask


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
    f"""<thinking>To solve 'Plan the best strategy for a chess endgame', I will analyze the given position and consider various tactical and strategic options. The position is as follows: White has a King on g1, Rook on c1, Pawns on a2, b3, e3, g2, h2; Black has a King on g8, Rook on f8, Pawns on a7, b6, e6, g7, h7. It's White to move.</thinking>
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
    f"""<thinking>To solve 'Write a short story about a robot learning to understand human emotions', I will create a narrative with specific characters and events that depict this journey. The main character will be a robot named Max in a modern human household.</thinking>
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
    f"""<thinking>To solve 'Design a sustainable urban transportation system', I will focus on the city of Greenfield, which faces traffic congestion and pollution due to heavy reliance on personal vehicles.</thinking>
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
    f"""<thinking>To solve 'Develop a comprehensive marketing strategy for the launch of a new eco-friendly water bottle', I will analyze the target market, competition, and unique selling propositions (USPs) of the product.</thinking>
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
    f"""<thinking>To solve 'Create a comprehensive lesson plan for teaching high school biology on the topic of cellular respiration', I will outline the objectives, materials needed, teaching methods, activities, and assessments to ensure effective learning.</thinking>
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
    f"""<thinking>To solve 'Design a mobile application for personal finance management aimed at millennials', I will analyze user needs, market trends, and essential features to create a user-friendly and effective app.</thinking>
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
    f"""<thinking>To solve 'Develop a comprehensive business plan for opening a new restaurant specializing in farm-to-table cuisine', I will analyze the market, define the restaurant's concept, plan operations, and outline financial projections.</thinking>
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
    f"""<thinking>To solve 'Design an effective user onboarding process for a new mobile fitness application', I will outline the user journey, key features to highlight, engagement strategies, and metrics to track success.</thinking>
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
    f"""<thinking>To solve 'Develop a Python script to analyze a sales dataset and generate insightful reports', I will outline the steps to import the data, clean it, perform exploratory data analysis (EDA), visualize key metrics, and export the results.</thinking>
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
        print("Dataset loaded successfully.")
        return data
    except FileNotFoundError:
        print("File not found.")
        return None

# Step 2: Inspect the dataset for missing values and perform data cleaning
def clean_data(data):
    print("Initial data snapshot:")
    print(data.head())
    print("\nMissing values per column:")
    print(data.isnull().sum())

    # Handling missing values
    data = data.dropna()  # Dropping rows with missing values for simplicity
    print("\nData after removing missing values:")
    print(data.isnull().sum())
    return data

# Step 3: Perform exploratory data analysis
def perform_eda(data):
    # Calculate total sales
    data['Total_Sales'] = data['Quantity_Sold'] * data['Unit_Price']
    total_sales = data['Total_Sales'].sum()
    print(f"Total Sales: ${{total_sales:,.2f}}")

    # Sales by Category
    sales_by_category = data.groupby('Category')['Total_Sales'].sum().sort_values(ascending=False)
    print("\nSales by Category:")
    print(sales_by_category)

    # Sales by Region
    sales_by_region = data.groupby('Region')['Total_Sales'].sum().sort_values(ascending=False)
    print("\nSales by Region:")
    print(sales_by_region)

    # Monthly Sales Trends
    data['Month'] = data['Date'].dt.to_period('M')
    monthly_sales = data.groupby('Month')['Total_Sales'].sum()
    print("\nMonthly Sales Trends:")
    print(monthly_sales)

    # Top 10 Products
    top_products = data.groupby('Product_Name')['Total_Sales'].sum().sort_values(ascending=False).head(10)
    print("\nTop 10 Products:")
    print(top_products)

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

    print("Visualizations created and saved.")

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
    print("Sales analysis report generated as 'Sales_Analysis_Report.pdf'.")

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


class PromptSuggestions(BaseModel):
    prompt_modifications: List[str] = Field(..., title="List of Prompt Suggestions")


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


class Reflection:
    content: str
    reward: float
    step_number: int


class Step:
    """
    A class representing a step in a process.

    Attributes:
        description (str): A text description of the step.
        step_number (int): The sequential number/position of this step.
        remaining_budget (int): The budget amount remaining at this step.
        reflections (List[Reflection]): A list of Reflection objects associated with this step.
    """

    description: str
    step_number: int
    remaining_budget: int
    reflection: Reflection


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
    answer: Optional[str] = None
    final_reward: Optional[float] = None
    plan = Optional[Plan] = None
    output_type: OutputType


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

    task = Task
    steps: List[Step] = Field(default_factory=list)
    reflections: List[Reflection] = Field(default_factory=list)
    answer: Optional[str] = None
    final_reward: Optional[float] = None


class CompletionStatus(BaseModel):
    completion: bool


class FinalReviewScore(BaseModel):
    score: float


class AdvancedPromptEngineer:
    def __init__(self, config: PromptEngineeringConfig):
        self.config = config
        self.knowledge_base = {}  # For Retrieval-Augmented Generation
        self.task_object = None

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

    def judge_step_completion(
        self, step: Step, plan_step: PlanStep
    ) -> Tuple[bool, int]:
        """
        Judge the completion status of a step based on whether it completes the intended action described in the plan step. An LLM evaluates the step's completion based on the alignment with the plan step and the task's requirements

        Args:
            step (Step): The step to evaluate.
            plan_step (PlanStep): The corresponding plan step.

        Returns:
            Tuple[bool, int]: A tuple containing a boolean indicating completion status and the number of the next plan step, based on the completion status.
        """
        prompt = f"""
        Evaluate the completion of the following Step based on the Plan Step:
        Plan Step Name: '{plan_step.step_name}'
        Plan Step Description: '{plan_step.step_description}'
        Plan Step Explanation: '{plan_step.step_explanation}'
        Plan Step Expected Output: '{plan_step.step_output}'
        Plan Step Full Text: '{plan_step.step_full_text}'

        Step: '{step.description}'

        Does the Step complete the intended action described in the Plan Step? Provide your answer as a boolean value (True/False) of whether the Step is complete according to the Plan Step.

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
            assert isinstance(output, bool)
            next_step_number = (
                plan_step.step_number + 1 if output else plan_step.step_number
            )
            return (output, next_step_number)
        except Exception as e:
            print(f"Error: {e}")
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
            print(f"Error: {e}")
            return (None, subtask.step_number)

    def judge_step(self, step: Step, task: Task) -> Reflection:
        """
        An LLM judges the quality of a step based on how well it aligns with the task, its clarity, relevance, completeness, correctness, and logical coherence.
        """
        prompt = f"""
        Evaluate the following step in the context of solving the task: '{task.description}'.
        Step:
        <step>{step.description}</step>
        <count>{step.remaining_budget}</count>
        <reflection>Provide a reflection on the step's quality, including its clarity, relevance, completeness, correctness, and logical coherence. Enclose your reflection within <reflection> tags.</reflection>
        <reward>Assign a quality score between 0.0 and 1.0 based on the reflection. Enclose the score within <reward> tags.</reward>
        """

        messages = [
            {
                "role": "system",
                "content": f"""You are an expert AI assistant tasked with evaluating the quality of problem-solving steps. Provide detailed reflections and assign quality scores based on the step's clarity, relevance, completeness, correctness, and logical coherence.
                Your feedback should be constructive, actionable, and aimed at improving the step's overall quality, focused only on the step and the task. Check for errors, flaws, or inconsistencies in the step. After providing your reflection inside <reflection> tags, assign a quality score between 0.0 and 1.0 using <reward> tags.
                Please encapsulate your reflection within <reflection> tags and assign a quality score between 0.0 and 1.0 using <reward> tags.
                """,
            },
            {"role": "user", "content": prompt},
        ]

        response = self.call_openai(messages=messages, temperature=0.2)

        reflection_match = re.search(
            r"<reflection>(.*?)<\/reflection>", response, re.DOTALL
        )
        reward_match = re.search(
            r"<reward>(0\.\d+|1\.0)<\/reward>", response, re.DOTALL
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
            print(f"Error retrieving external information: {e}")
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
        internal_info = self.knowledge_base.get(task, "")
        external_info = self.retrieve_external_info(task)
        return f"{internal_info}\n{external_info}"

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
        self, interaction: Interaction, task: str
    ) -> Interaction:
        """
        Explores alternative solutions if confidence is low based on final_reward.
        """
        if (
            interaction.final_reward
            and interaction.final_reward < self.config.confidence_thresholds[1]
        ):
            if self.config.backtrack:
                prompt = f"Take a different approach to solve the following task.\n\nTask: {task}\n"
                messages = [
                    {
                        "role": "system",
                        "content": "You are an AI assistant tasked with solving complex problems. You should explore alternative approaches to find the best solution, always thinking critically and reasoning through the problem step by step.",
                    },
                    {"role": "user", "content": prompt},
                ]
                new_response = self.call_openai(
                    messages=messages,
                    temperature=min(self.config.temperature * 2, 1.0),
                )
                if new_response:
                    new_interaction = self.parse_response(new_response)
                    if (
                        new_interaction.final_reward
                        and new_interaction.final_reward > interaction.final_reward
                    ):
                        print(
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
        task: str,
        prompt: str,
        existing_interaction: Interaction,
        output_type: OutputType,
    ) -> list[Interaction]:
        """
        Generates responses from multiple agents for collaborative reasoning.
        """
        # TODO: Use conversation_manager.py for more advanced multi-agent interactions
        type_of_output = output_type.output_type
        ext_of_output = output_type.file_extension
        agent_responses = list[Interaction]
        tag_explainer = f"""1. Begin by enclosing all thoughts within <thinking> tags, exploring multiple angles and approaches. You will use these as your mental scratchpad to brainstorm and consider various strategies, as well as to step through the problem-solving process. The current step involves outlining the next logical action within <step> tags, focusing on the immediate next step in the problem-solving process.
2. The <count> tag will help you keep track of the remaining steps and budget your actions accordingly.
3. Break down the solution into clear steps within <step> tags. These steps should be concise and focused on the next specific action to take in the problem-solving process after the previous <step> tags.
4. Use the <thinking> tags as a scratchpad to write out all calculations and reasoning explicitly, and the <step> tags to outline your best resulting answer for the next step in the task, after the previous <step> tags (the count will be decremented after each step and enclosed in <count> tags).
5. For each step, after thinking inside <thinking> tags, you will provide a clear, concise response within <step> tags, outlining only the next step in the problem-solving process that comes after the most recent <step> tags.
6. You will be regularly evaluated on your progress by an external reviewer inside <reflection> tags. They will be critical and honest about your reasoning process.
7. The reviewer will also assign a quality score between 0.0 and 1.0 using <reward> tags after each reflection. Use this to guide your approach:
8. Please wrap your final responses in the following tags: <agent_response>...</agent_response>. Synthesize the final answer to the main problem/task within <agent_response> tags, providing a clear, concise final output, and only if the task has been solved successfully.
9. This will conclude with a final review in <reflection> tags on the overall solution, discussing effectiveness, challenges, and solutions. A final reward score of 0.0 to 1.0 will be assigned based on the quality of the solution, using <final_reward> tags.

You will be stopped when the count enclosed in <count> tags reaches 0. 
Please start at the most recent step and continue from there. 
Example:
{examples_for_prompt_instructions[random.randint(0, len(examples_for_prompt_instructions) - 1)]}

Now, let's solve the following task, focusing specifically on the <thinking>, and <step> tags, and focusing ONLY on the current step, generating <thinking> and <step> tags for the next step in the task after the most recent <step> tag, building directly on the previous steps. You will be stopped after you complete another <step> tag, or <answer> tag, or when the <count> reaches 0.
Always think in <thinking> tags before generating a <step> tag, and ensure that it builds on the previous steps.
Remember to provide a clear and concise answer within <answer> tags at the end of the process, if the task is completed within the step budget. Also, the expected format of the end result the user is looking for is {type_of_output} file extension {ext_of_output}, so keep that in mind for the last step.

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

            messages = [
                {
                    "role": "system",
                    "content": agent_intro_mapping[i]["intro"].join(
                        tag_explainer.split("Example:")[0]
                        + "\n"
                        + examples_for_prompt_instructions[
                            random.randint(0, len(examples_for_prompt_instructions) - 1)
                        ]
                    ),
                },
                {
                    "role": "user",
                    "content": agent_intro_mapping[i]["template"].format(task=task)
                    + "\n\n"
                    + prompt,
                },
            ]
            response = self.call_openai(
                messages=messages,
                temperature=agent_intro_mapping[i]["temperature"],
                stop=["</agent_response>"],
            )
            if response:
                response_interaction = self.parse_response(
                    response,
                    task,
                    steps_objs=existing_interaction.steps,
                    reflections_objs=existing_interaction.reflections,
                    interaction=existing_interaction,
                )
                assert isinstance(response_interaction, Interaction)
                agent_responses.append(response_interaction)
        print("Agent Interactions:", agent_responses)
        assert isinstance(agent_responses, list[Interaction])
        return agent_responses

    def collaborative_reasoning_main(
        self,
        task: str,
        prompt: str,
        existing_interaction: Interaction,
        output_type: OutputType,
    ) -> Interaction:
        """
        Aggregates responses from multiple agents and selects the best one based on reward scores.
        """
        # Collaborative Multi-Agent Reasoning
        # TODO: Use conversation_manager.py for more advanced multi-agent interactions
        agent_responses = list[Interaction]
        agent_responses = self.collaborative_reasoning(
            task, prompt, existing_interaction, output_type
        )
        # Select the interaction with the highest final_reward
        best_interaction: Interaction = max(
            agent_responses, key=lambda x: x.final_reward if x.final_reward else 0.0
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
            print(f"Error: {e}")
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
            print("Refining prompt by increasing step budget due to low confidence.")
            for reflection in interaction.reflections:
                if reflection.reward < 0.5:
                    prompt += f"\n<thinking>Reflection: {reflection.content}</thinking>"
            return prompt
        return prompt

    def condense_plan(self, plan: Plan) -> str:
        """
        Condenses the plan into a concise summary.
        """
        condensed_plan = f"Plan Summary:\n\n{plan.description}\n\nSteps:\n"
        for step in plan.steps:
            condensed_plan += f"- Step {step.step_number}: {step.step_name}\n"
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
   - **<thinking>**: Enclose all internal thoughts, explorations, and considerations within `<thinking>` tags. This section should explore multiple angles and approaches to the problem.
   - **<step>**: Break down the solution into clear, actionable steps within `<step>` tags.
   - **<count>**: After each `<step>`, use a `<count>` tag to indicate the remaining budget (number of steps left).
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

- **Thinking**: `<thinking>` tags for brainstorming.
- **Steps**: `<step>` tags for actionable steps.
- **Count**: `<count>` tags showing remaining budget.
- **Reflection**: `<reflection>` tags for step evaluation.
- **Reward**: `<reward>` tags for scoring.
- **Answer**: `<answer>` tags for final output.
- **Final Reward**: `<final_reward>` for overall quality scoring.

**Example Response Structure:**

**Task Execution:**
- When presented with a task, follow the structured tagging and step-by-step approach as outlined.
- Ensure all sections are properly enclosed within their respective tags.
- Maintain clarity, thoroughness, and adaptability throughout the problem-solving process.

By adhering to these guidelines, you will provide structured, transparent, and high-quality solutions that are easy to follow and evaluate.
"""
        prompt = f"""Begin by enclosing all thoughts within <thinking> tags, exploring multiple angles and approaches. You will use these as your mental scratchpad to brainstorm and consider various strategies, as well as to step through the problem-solving process.
Break down the solution into clear steps within <step> tags. Start with a {adjusted_budget}-step budget, requesting more for complex problems if needed.
There will be <count> tags after each step to show the remaining budget. You will be stopped after the value reaches 0.
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
        for attempt in range(self.config.max_retries):
            try:
                response = openai.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stop=stop_sequence,
                )

                return response.choices[0].message["content"]
            except Exception as e:
                print(f"Unexpected error: {e}.")
                break
        return ""

    # -------------------------------
    # Response Parsing
    # -------------------------------
    def parse_response(
        self,
        response: str,
        task: str,
        steps_objs: List[Step] = None,
        reflections_objs: List[Reflection] = None,
        current_step_number: int = 0,
        current_remaining_budget: int = 0,
        interaction: Interaction = None,
        initial_budget: int = 0,
    ) -> Interaction:
        """
        Parses the OpenAI API response to extract steps, reflections, answers, and rewards.
        """
        # TODO: Implement a more robust response parsing mechanism, like using structured output from the model
        if interaction is None:
            interaction = Interaction()
        # Extract steps
        steps = re.findall(r"<step>(.*?)<\/step>", response, re.DOTALL)
        first_count = re.search(
            r"<count>(\d+)<\/count>", response
        )  # Represents the initial step budget
        if first_count:
            first_count = int(first_count.group(1))
        else:
            first_count = 0
        if initial_budget != 0:
            if first_count != initial_budget:
                first_count = initial_budget
                print("Initial budget mismatch. Adjusting initial budget.")

        counts = re.findall(r"<count>(\d+)<\/count>", response)
        # Extract reflections
        # Revert reflections to the original pattern
        reflections = re.findall(
            r"<reflection>(.*?)</reflection>",
            response,
            re.DOTALL,
        )

        # Use the modified pattern for rewards
        rewards = re.findall(
            r"</reflection>\s*.*?<reward>(0\.\d+|1\.0)</reward>(?=.*(?=<reflection>|<step>))",
            response,
            re.DOTALL,
        )
        i = 0

        if steps_objs is not None:
            if len(steps_objs) != len(steps):
                if len(steps_objs) > len(steps):
                    print("Steps mismatch. Adjusting steps.")
                    missing_steps = {}
                    for step in steps_objs:
                        if step.description.strip() not in [s.strip() for s in steps]:
                            missing_steps.update(
                                {step.description.strip(): step.step_number}
                            )

                    # Ensure lists are same length before processing
                    while len(reflections) < len(steps_objs):
                        reflections.append(None)
                    while len(rewards) < len(steps_objs):
                        rewards.append(None)

                    for step in steps_objs:
                        if step.description.strip() in missing_steps.keys():
                            step_num = missing_steps[step.description.strip()]

                            # Validate step number is sequential
                            if step_num > 0 and not any(
                                s.step_number == step_num - 1 for s in steps_objs
                            ):
                                raise ValueError(
                                    f"Non-sequential step number found: {step_num}"
                                )

                            # Insert step and its attributes
                            steps.insert(step_num, step.description.strip())
                            counts.insert(step_num, step.count)

                            if hasattr(step, "reflection"):
                                reflections[step_num] = step.reflection
                            if hasattr(step, "reward"):
                                # Handle final reward for last step
                                if step_num == len(steps_objs) - 1:
                                    rewards[step_num] = (
                                        f"<final_reward>{step.reward}</final_reward>"
                                    )
                                else:
                                    rewards[step_num] = (
                                        f"<reward>{step.reward}</reward>"
                                    )

                    # Verify each step has required attributes
                    for i in range(len(steps)):
                        if reflections[i] is None or rewards[i] is None:
                            raise ValueError(
                                f"Missing reflection or reward for step {i}"
                            )

        def find_next_reflection(text, current_position):
            step_tag = re.compile(r"<step>.*?</step>", re.DOTALL)
            reflection_tag = re.compile(r"<reflection>(.*?)</reflection>", re.DOTALL)

            # Search for step tags before the current position
            if step_tag.search(text[current_position:]):
                return -1  # Indicating another step tag found before reflections

            # Find the next reflection tag after the current position
            match = reflection_tag.search(text[current_position:])

            if match:
                return match.group(1).strip()  # Return the content between the tags

            return -1  # Indicating no reflections tag found no reflections tag found

        for step_desc, count in zip(steps, counts):
            this_step = None
            if isinstance(count, str):
                count = int(count)
            if not isinstance(step_desc, str):
                step_desc = step_desc.group(1)
            step_reflection = None
            if len(reflections) == len(steps):
                step_reflection = reflections[i]
            else:
                step_reflection = find_next_reflection(
                    response, response.find(steps[i]) + len(steps[i])
                )
                if step_reflection == -1:
                    # First, check if this step exists in steps_objs
                    if steps_objs and isinstance(step_desc, str):
                        for step_obj in steps_objs:
                            if isinstance(step_obj, Step):
                                if (
                                    step_obj.step_number
                                    == int(first_count) - int(count)
                                    or step_obj.description.strip().lower()
                                    == step_desc.strip().lower()
                                ):
                                    step_reflection = step_obj.reflection
                                    this_step = step_obj
                                    break
                    elif reflections_objs and isinstance(step_desc, str):
                        for reflection_obj in reflections_objs:
                            if isinstance(reflection_obj, Reflection):
                                if (
                                    reflection_obj.step_number
                                    == int(first_count) - int(count)
                                    and not steps_objs
                                ):
                                    step_reflection = reflection_obj
                                    break
                                elif steps_objs:
                                    for step_obj in steps_objs:
                                        if (
                                            step_obj.step_number
                                            == reflection_obj.step_number
                                        ):
                                            step_reflection = reflection_obj
                                            this_step = step_obj
                                            break
                    if this_step is None:
                        this_step = Step(
                            description=step_desc,
                            remaining_budget=int(count),
                            step_number=int(first_count) - int(count),
                            reflection=None,
                        )

                    if step_reflection is None:
                        step_reflection = self.judge_step(this_step, task)
                    this_step.reflection = step_reflection
                    interaction.steps.append(this_step)
                else:
                    if isinstance(step_reflection, str):
                        # Check for <reward> tag after reflection closing tag, but only if it's not the last reflection and only if it's not followed by another reflection or step tag
                        # Extract the substring starting after step_reflection
                        substring = response[
                            response.find(step_reflection) + len(step_reflection) :
                        ]

                        # Search for <reward> tag ensuring it's not followed by <reflection> or <step> tags
                        if i < len(reflections) - 1 and len(rewards) > i:
                            step_reward = re.search(
                                r"</reflection>\s*.*?<reward>(0\.\d+|1\.0)<\/reward>(?!.*(<reflection>|<step>))",
                                substring,
                                re.DOTALL,
                            )
                            step_rew_float = (
                                float(step_reward.group(1)) if step_reward else 0.0
                            )

                        step_reflection_obj = Reflection(
                            content=step_reflection.strip(), reward=step_rew_float
                        )
                        if steps_objs and isinstance(step_desc, str):
                            for step_obj in steps_objs:
                                if (
                                    step_obj.step_number
                                    == int(first_count) - int(count)
                                    or step_obj.description.strip().lower()
                                    == step_desc.strip().lower()
                                ):
                                    step_obj.reflection = step_reflection_obj
                                    this_step = step_obj

                                    break

                        if this_step is None or not isinstance(this_step, Step):
                            this_step = Step(
                                description=step_desc.strip(),
                                remaining_budget=int(count),
                                step_number=int(first_count) - int(count),
                                reflection=step_reflection_obj,
                            )
                        interaction.steps.append(this_step)
                    elif isinstance(step_reflection, Reflection):
                        if steps_objs and isinstance(step_desc, str):
                            for step_obj in steps_objs:
                                if (
                                    step_obj.step_number
                                    == int(first_count) - int(count)
                                    or step_obj.description.strip().lower()
                                    == step_desc.strip().lower()
                                ):
                                    step_obj.reflection = step_reflection
                                    this_step = step_obj
                                    break
                        if this_step is None:
                            this_step = Step(
                                description=step_desc.strip(),
                                remaining_budget=int(count),
                                step_number=int(first_count) - int(count),
                                reflection=step_reflection,
                            )
                        interaction.steps.append(this_step)
                    elif reflections_objs and isinstance(step_desc, str):
                        for reflection_obj in reflections_objs:
                            if isinstance(reflection_obj, Reflection):
                                if (
                                    reflection_obj.step_number
                                    == int(first_count) - int(count)
                                    and not steps_objs
                                ):
                                    reflection_obj.content = step_reflection
                                    this_step = reflection_obj
                                    break
                                elif steps_objs:
                                    for step_obj in steps_objs:
                                        if (
                                            step_obj.step_number
                                            == reflection_obj.step_number
                                        ):
                                            reflection_obj.content = step_reflection
                                            this_step = reflection_obj
                                            break
                        if this_step is None:
                            this_step = Reflection(
                                content=step_reflection.strip(),
                                reward=0.0,
                                step_number=int(first_count) - int(count),
                            )
                        interaction.reflections.append(this_step)
            i += 1

        for reflection, reward in zip(reflections, rewards):
            # Check if the reflection is already in the interaction object
            found = False
            if isinstance(reflection, str):
                for ref in interaction.reflections:
                    if ref.content == reflection:
                        found = True
                        ref.reward = float(reward) if ref.reward == 0.0 else ref.reward
                        break
            elif isinstance(reflection, Reflection):
                for ref in interaction.reflections:
                    if ref.content == reflection.content:
                        found = True
                        ref.reward = float(reward) if ref.reward == 0.0 else ref.reward
                        break
            if not found:
                if isinstance(reflection, str):
                    interaction.reflections.append(
                        Reflection(content=reflection, reward=float(reward))
                    )
                elif isinstance(reflection, Reflection):
                    interaction.reflections.append(reflection)

        # Extract answer
        answer_match = re.search(r"<answer>(.*?)<\/answer>", response, re.DOTALL)
        if answer_match:
            interaction.answer = answer_match.group(1).strip()

        # Extract final reward
        final_reward_match = re.search(
            r"<final_reward>(0\.\d+|1\.0)<\/final_reward>", response, re.DOTALL
        )
        if final_reward_match:
            interaction.final_reward = float(final_reward_match.group(1))
        assert isinstance(interaction, Interaction)
        return interaction

    # -------------------------------
    # Self-Consistency
    # -------------------------------
    def self_consistency(
        self, task: Task, prompt: str, existing_interaction: Interaction
    ) -> Interaction:
        """
        Implements Self-Consistency by generating multiple reasoning paths and selecting the most consistent one.
        """
        responses = List[Interaction]
        steps_objs = existing_interaction.steps if existing_interaction else None
        reflections_objs = (
            existing_interaction.reflections if existing_interaction else None
        )

        for _ in range(self.config.n if self.config.n > 4 else 5):
            temperature = random.uniform(0.0, 0.5 + (min(0.1 * self.config.n, 1.2)))
            response = self.call_openai(prompt, temperature=temperature)
            if response:
                interaction = self.parse_response(
                    response,
                    task.refined_description,
                    steps_objs,
                    reflections_objs,
                    interaction=existing_interaction,
                )
                responses.append(interaction)
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

        return response.choices[0].message["content"].strip()

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
        pattern = re.compile(r"<prompt>(.*?)</prompt>", re.DOTALL)
        match = pattern.search(response.choices[0].message["content"])
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

    #     return response.choices[0].message["content"].strip()

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
        merged_interaction = Interaction()
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
            print(f"Error in merging interactions: {e}")
            raise RuntimeError(f"Error in merging interactions: {e}")

    def main(self, task: str) -> Interaction:
        """
        Integrates all components to process the task and generate the final answer.
        """
        # Step 1: Retrieve Information
        retrieved_info = self.retrieve_information(task)
        print("Retrieved Information:\n", retrieved_info)
        interaction = Interaction()
        refined_task = self.task_into_prompt(task)
        print("Refined Task:", refined_task)

        task_object = Task()
        task_object.description = task
        task_object.refined_description = refined_task
        task_object.output_type = output_type_determination(task)

        # Step 2: Assess Complexity
        complexity, plan = self.assess_complexity(refined_task)
        print("Assessed Complexity:", complexity)
        assert complexity > 0.0, "Complexity assessment failed."
        assert isinstance(plan, Plan), "Complexity plan generation failed."

        initial_step_budget = len(plan.steps)
        subtasks = []
        for step in plan.steps:
            subtasks.append(step.subtasks)

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
        print("Adjusted Step Budget:", adjusted_budget)

        # Step 4: Generate Initial Prompt
        initial_prompt, system_prompt = self.generate_initial_prompt(
            refined_task, retrieved_info, adjusted_budget, complexity
        )
        print("System Prompt:\n", system_prompt)
        print("Initial Prompt:\n", initial_prompt)
        current_prompt = initial_prompt

        # Find the step that has step_number = 1
        current_plan_step = next((x for x in plan.steps if x.step_number == 1), None)
        current_plan_subtask = None
        if current_plan_step is not None and current_plan_step.subtasks:
            current_plan_subtask = next(
                (x for x in current_plan_step.subtasks if x.subtask_number == 1), None
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
            if current_plan_step and isinstance(current_plan_step, PlanStep):
                curr_num = current_plan_step.step_number
                curr_name = current_plan_step.step_name
                curr_desc = current_plan_step.step_description
                curr_explanation = current_plan_step.step_explanation
                curr_output = current_plan_step.step_output
                curr_fulltext = current_plan_step.step_full_text
            else:
                raise ValueError("Invalid type for current_plan_step")
            if current_plan_subtask and isinstance(current_plan_subtask, Subtask):
                curr_st_num = current_plan_subtask.subtask_number
                curr_st_name = current_plan_subtask.subtask_name
                curr_st_desc = current_plan_subtask.subtask_description
                curr_st_explanation = current_plan_subtask.subtask_explanation
                curr_st_output = current_plan_subtask.subtask_output
                curr_st_fulltext = current_plan_subtask.subtask_full_text

            condensed_plan = self.condense_plan(plan)
            step_prompt = f"""Suggested Plan:
{condensed_plan}

For the current step, focus on the following plan step or plan sub-task:
- Step Number: {curr_num}
- Step Name: {curr_name}
- Step Description: {curr_desc}
- Step Explanation: {curr_explanation}
- Step Expected Output: {curr_output}
- Step Full Text: {curr_fulltext}

"""
        st_prompt = ""
        if curr_st_num is not None and current_plan_subtask is not None:
            st_prompt = f"""Specifically, for the current sub-task, focus on the following plan sub-task of step {curr_num}:
- Sub-Task Number: {curr_st_num}
- Sub-Task Name: {curr_st_name}
- Sub-Task Description: {curr_st_desc}
- Sub-Task Explanation: {curr_st_explanation}
- Sub-Task Expected Output: {curr_st_output}
- Sub-Task Full Text: {curr_st_fulltext}
"""
            custom_init_prompt = initial_prompt + f"\n{step_prompt}"
            if st_prompt is not None and st_prompt != "":
                custom_init_prompt += f"\n{st_prompt}"

            current_prompt.replace(initial_prompt, custom_init_prompt)

            count = adjusted_budget - step
            # Add a count tag to the prompt
            current_prompt += f"\n<count>{count}</count>\n"
            # add first thinking tag
            current_prompt += f"\n<thinking>"

            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": current_prompt, "name": "Human User"},
            ]
            # Step 5: Collaborative Multi-Agent Reasoning
            # for collaborative_reasoning, input only the current prompt minus the initial prompt
            agent_prompt = current_prompt.replace(initial_prompt, "")
            if self.config.agents > 0:
                agent_interaction = self.collaborative_reasoning_main(
                    task, agent_prompt, interaction, task_object.output_type
                )
            # Step 6: Self-Consistency Check
            self_consistent_interaction = self.self_consistency(
                task_object, current_prompt, interaction
            )
            print("Self-Consistent Interaction:", self_consistent_interaction)

            # Step 7: Aggregate and Select Best Interaction
            all_interactions = []
            all_interactions.append(agent_interaction)
            all_interactions.append(self_consistent_interaction)
            best_interaction = max(
                all_interactions,
                key=lambda x: x.final_reward if x.final_reward else 0.0,
            )

            # Step 8: Dynamic Confidence Exploration
            if best_interaction.final_reward < self.config.confidence_thresholds[1]:
                best_interaction = self.dynamic_confidence_exploration(
                    best_interaction, task, current_prompt
                )

            # Step 9: Prompt Refinement
            if best_interaction.final_reward < self.config.confidence_thresholds[2]:
                refined_prompt = self.refine_prompt(best_interaction, task)
                best_interaction = self.self_consistency(task, refined_prompt)

            # Now, judge whether the latest plan step or sub-task was completed successfully
            assert isinstance(best_interaction, Interaction)
            assert isinstance(interaction, Interaction)
            interaction = self.merge_interactions(interaction, best_interaction)
            assert isinstance(interaction, Interaction)

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
                else:
                    complete, next_step_num = self.judge_step_completion(
                        best_interaction.steps[-1], current_plan_step
                    )
                    if complete:
                        current_plan_step.completed = True
                    else:
                        current_plan_step.completed = False
                if (
                    current_plan_step.completed
                    and current_plan_step.step_number < len(plan.steps)
                    and current_plan_step.step_number
                    < max(plan.steps, key=lambda x: x.step_number).step_number
                ):
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
                                if x.subtask_number == 1
                            ),
                            None,
                        )
                    else:
                        current_plan_subtask = None
            if interaction.answer and interaction.answer != "":
                interaction = self.judge_final_answer(task, interaction)
                return interaction

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
            print("Adjusting step budget due to high complexity.")
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
Objective: Develop and implement a Python script for a custom strategy board game called "Resource Wars." The script must accomplish the following tasks:

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

    # TODO: Replace the task_description with steps from the plans generated by complexity_measures.py
    result = engineer.main(task_description)
    print("\nFinal Answer:")
    print(result.answer)
    print("\nFinal Reflection:")
    print(f"Reward Score: {result.final_reward}")
