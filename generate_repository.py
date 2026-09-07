"""
generate_repository.py
------------------------
Creates the bundled reference document repository used by the
Plagiarism & Duplicate Scanner. This runs ONCE to populate the
`Reference_Topics/` folder with sample documents so that TF-IDF and PCA
have real data to fit on -- no external downloads required.

Run this only if you want to regenerate/reset the repository, or
if you want to add your own topics to expand it.
"""

import os

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reference_Topics")

DOCUMENTS = {
    "climate_change_a.txt": (
        "Climate change refers to long-term shifts in temperatures and weather "
        "patterns, primarily driven by human activities such as burning fossil "
        "fuels. Since the industrial revolution, the burning of coal, oil, and "
        "gas has released large amounts of greenhouse gases into the atmosphere, "
        "trapping heat and causing global temperatures to rise. Effects include "
        "melting glaciers, rising sea levels, more frequent extreme weather "
        "events, and disruptions to ecosystems worldwide. Scientists agree that "
        "reducing carbon emissions through renewable energy adoption, "
        "reforestation, and sustainable practices is essential to limit further "
        "warming. International agreements like the Paris Accord aim to unite "
        "countries in reducing greenhouse gas output and adapting to unavoidable "
        "changes already underway."
    ),
    "climate_change_b.txt": (
        "Global warming describes the gradual increase in Earth's average "
        "temperature and the resulting changes in climate patterns, largely "
        "caused by human industrial activity. The widespread use of fossil "
        "fuels like coal, oil, and natural gas since the 1800s has pumped "
        "enormous quantities of heat-trapping gases into our atmosphere. As a "
        "result, we are witnessing shrinking ice caps, higher ocean levels, an "
        "increase in severe storms and heatwaves, and damage to natural "
        "habitats across the planet. Most climate scientists believe that "
        "switching to clean energy sources, planting more trees, and adopting "
        "greener lifestyles can help slow this warming trend. Global "
        "cooperation efforts, such as the Paris Climate Agreement, seek to "
        "bring nations together to cut emissions and prepare for the changes "
        "that are already happening."
    ),
    "ai_healthcare_a.txt": (
        "Artificial intelligence is transforming the healthcare industry by "
        "enabling faster and more accurate diagnoses. Machine learning "
        "algorithms can analyze medical images such as X-rays and MRIs to "
        "detect diseases like cancer at early stages, often outperforming "
        "human radiologists in specific tasks. AI-powered systems also help "
        "predict patient risks, personalize treatment plans, and streamline "
        "administrative tasks like scheduling and billing. Hospitals are "
        "increasingly adopting AI chatbots to handle patient inquiries and "
        "virtual assistants to support clinical decision-making. Despite these "
        "advances, challenges remain around data privacy, algorithmic bias, "
        "and the need for regulatory oversight to ensure patient safety and "
        "trust in these technologies."
    ),
    "ai_healthcare_b.txt": (
        "Healthcare is being reshaped by advances in artificial intelligence, "
        "which allow for quicker and more precise medical diagnoses. "
        "Algorithms trained on large datasets can review scans such as X-rays "
        "and MRI images to identify conditions like tumors early on, sometimes "
        "with accuracy that rivals or exceeds trained doctors. These "
        "intelligent systems also assist in forecasting patient health risks, "
        "tailoring individual treatment strategies, and automating routine "
        "office work such as appointment booking and invoicing. Many medical "
        "centers now use AI-driven virtual assistants to answer patient "
        "questions and support doctors in making clinical choices. However, "
        "concerns about data security, unfair bias in algorithms, and the "
        "necessity of proper regulation continue to be important issues for "
        "building confidence in these tools."
    ),
    "exercise_benefits_a.txt": (
        "Regular physical exercise offers numerous benefits for both physical "
        "and mental health. Engaging in activities like walking, running, "
        "swimming, or strength training helps strengthen the heart, improve "
        "lung capacity, and maintain a healthy body weight. Exercise also "
        "releases endorphins, chemicals in the brain that reduce stress and "
        "improve mood, making it an effective tool against anxiety and "
        "depression. Additionally, consistent physical activity can lower the "
        "risk of chronic diseases such as diabetes, heart disease, and certain "
        "cancers. Health experts generally recommend at least 150 minutes of "
        "moderate exercise per week, combined with muscle-strengthening "
        "activities, to achieve optimal health outcomes."
    ),
    "exercise_benefits_b.txt": (
        "Working out on a consistent basis provides a wide range of "
        "advantages for the body and mind. Activities such as jogging, "
        "cycling, swimming, or lifting weights can boost cardiovascular "
        "strength, enhance breathing capacity, and help people manage their "
        "weight effectively. Exercise triggers the release of endorphins, "
        "natural brain chemicals known to ease stress and lift mood, which is "
        "why staying active is often recommended for managing anxiety and "
        "depression. Furthermore, staying physically active regularly can "
        "reduce the likelihood of developing long-term illnesses like "
        "diabetes, cardiovascular disease, and some forms of cancer. Medical "
        "professionals typically suggest getting at least two and a half "
        "hours of moderate-intensity exercise weekly, along with strength "
        "training, for the best health results."
    ),
    "french_revolution_a.txt": (
        "The French Revolution began in 1789 and dramatically transformed "
        "France's political and social structure. Fueled by widespread "
        "economic hardship, unfair taxation, and resentment toward the "
        "monarchy's absolute power, the revolution led to the overthrow of "
        "King Louis XVI and the eventual establishment of a republic. Key "
        "events included the storming of the Bastille, the Declaration of the "
        "Rights of Man, and the Reign of Terror, during which thousands were "
        "executed, including the king himself. The revolution's ideals of "
        "liberty, equality, and fraternity influenced political movements "
        "across the world and marked the decline of feudal privileges in "
        "Europe."
    ),
    "french_revolution_b.txt": (
        "Starting in 1789, the French Revolution brought sweeping changes to "
        "France's government and society. Driven by severe economic "
        "struggles, unjust tax policies, and growing frustration with the "
        "unchecked authority of the monarchy, the uprising resulted in the "
        "fall of King Louis XVI and the creation of a republican government. "
        "Notable moments included the fall of the Bastille prison, the "
        "adoption of the Declaration of the Rights of Man, and the Reign of "
        "Terror, a period when thousands lost their lives to execution, "
        "including the former king. The revolutionary principles of liberty, "
        "equality, and brotherhood inspired political change throughout the "
        "world and contributed to the erosion of feudal privileges across "
        "Europe."
    ),
    "renewable_energy_a.txt": (
        "Renewable energy sources, such as solar, wind, hydroelectric, and "
        "geothermal power, are becoming increasingly important as the world "
        "seeks alternatives to fossil fuels. Unlike coal, oil, and natural "
        "gas, renewable sources produce little to no greenhouse gas emissions "
        "and rely on naturally replenishing resources like sunlight and wind. "
        "Advances in technology have significantly lowered the cost of solar "
        "panels and wind turbines, making renewable energy more accessible "
        "and competitive with traditional power generation. Many countries "
        "are investing heavily in renewable infrastructure to reduce their "
        "carbon footprint, enhance energy security, and combat climate "
        "change, though challenges around energy storage and grid "
        "integration still need to be addressed."
    ),
    "renewable_energy_b.txt": (
        "As the world looks for substitutes to fossil fuel dependency, "
        "renewable energy options like solar, wind, hydropower, and "
        "geothermal energy are gaining significant traction. In contrast to "
        "coal, oil, and natural gas, these clean energy sources generate "
        "minimal to zero greenhouse gases and depend on naturally recurring "
        "resources such as sunshine and wind currents. Technological "
        "improvements have dramatically reduced the price of solar panels "
        "and wind turbines, making clean power generation increasingly "
        "affordable and able to compete with conventional energy methods. "
        "Numerous nations are pouring investment into renewable energy "
        "infrastructure to shrink their carbon emissions, strengthen energy "
        "independence, and fight against climate change, although issues "
        "related to storing energy and integrating it into power grids "
        "remain to be solved."
    ),
    "water_cycle_a.txt": (
        "The water cycle describes the continuous movement of water on, "
        "above, and below the surface of the Earth. It begins with "
        "evaporation, where heat from the sun turns water from oceans, "
        "rivers, and lakes into water vapor that rises into the atmosphere. "
        "This vapor cools and condenses to form clouds, eventually falling "
        "back to Earth as precipitation in the form of rain, snow, or hail. "
        "Once on the ground, water either flows into rivers and oceans as "
        "runoff, soaks into the soil as infiltration, or is absorbed by "
        "plants and released back into the air through transpiration. This "
        "cycle is essential for distributing fresh water across the planet "
        "and sustaining all forms of life."
    ),
    "water_cycle_b.txt": (
        "The hydrological cycle refers to the ongoing circulation of water "
        "across, above, and beneath the Earth's surface. The process starts "
        "with evaporation, as sunlight heats water in oceans, rivers, and "
        "lakes, converting it into vapor that ascends into the sky. This "
        "water vapor then cools down and condenses into clouds, which later "
        "release the water back to the ground as rain, snow, or hail. After "
        "reaching the surface, water can travel into rivers and oceans "
        "through runoff, seep into the ground through infiltration, or be "
        "taken up by plants and returned to the atmosphere via "
        "transpiration. This continuous loop plays a crucial role in "
        "spreading fresh water throughout the planet and supporting all "
        "living organisms."
    ),
    "remote_work_a.txt": (
        "Remote work has become increasingly common, especially following "
        "the global shift triggered by the COVID-19 pandemic. Many companies "
        "discovered that employees could remain productive while working "
        "from home, leading to a lasting change in workplace culture. "
        "Benefits of remote work include greater flexibility, reduced "
        "commuting time, and improved work-life balance for employees, while "
        "employers can save on office space and access a wider talent pool "
        "unrestricted by geography. However, remote work also presents "
        "challenges such as feelings of isolation, difficulties in team "
        "collaboration, and the need for strong self-discipline. As a "
        "result, many organizations have adopted hybrid models that combine "
        "in-office and remote work to balance these trade-offs."
    ),
    "remote_work_b.txt": (
        "The rise of remote work has accelerated significantly since the "
        "worldwide disruption caused by the COVID-19 pandemic. Numerous "
        "businesses realized that staff could stay just as productive while "
        "working outside a traditional office, resulting in a permanent "
        "shift in how workplaces operate. Advantages of working remotely "
        "include increased scheduling flexibility, less time spent "
        "commuting, and a better balance between personal and professional "
        "life, while companies benefit from lower office costs and the "
        "ability to hire talented individuals regardless of location. That "
        "said, remote work also brings certain difficulties, including "
        "feelings of loneliness, challenges coordinating with team members, "
        "and the requirement for solid personal discipline. Consequently, "
        "many organizations have shifted toward hybrid arrangements that "
        "blend office attendance with remote flexibility to manage these "
        "compromises."
    ),
    "blockchain_a.txt": (
        "Blockchain technology is a decentralized digital ledger system that "
        "records transactions across a network of computers in a way that is "
        "transparent and difficult to alter. Each block in the chain "
        "contains a set of transactions, and once added, it is "
        "cryptographically linked to the previous block, making tampering "
        "extremely difficult without altering every subsequent block. "
        "Originally developed as the foundation for cryptocurrencies like "
        "Bitcoin, blockchain has since found applications in supply chain "
        "management, voting systems, and secure record-keeping across "
        "various industries. Its main advantages include increased "
        "transparency, reduced reliance on central authorities, and "
        "enhanced security, although concerns about scalability and high "
        "energy consumption remain ongoing challenges for widespread "
        "adoption."
    ),
    "blockchain_b.txt": (
        "Blockchain is a distributed digital record-keeping system that logs "
        "transactions across a network of interconnected computers in a "
        "manner that is open and highly resistant to modification. Every "
        "block within the chain holds a group of transactions, and after "
        "being added, it becomes cryptographically connected to the "
        "preceding block, which makes altering the data nearly impossible "
        "without changing all following blocks as well. First created as "
        "the underlying infrastructure for cryptocurrencies such as "
        "Bitcoin, this technology has expanded into uses like tracking "
        "supply chains, enabling secure voting, and maintaining tamper-proof "
        "records across many different sectors. Key benefits include "
        "greater transparency, less dependency on centralized institutions, "
        "and improved security, though issues surrounding scalability and "
        "substantial energy usage continue to pose challenges for broader "
        "adoption."
    ),
    "internet_history.txt": (
        "The internet originated from a project called ARPANET, developed "
        "by the United States Department of Defense in the late 1960s to "
        "enable communication between research institutions. Over the "
        "following decades, key innovations such as the development of "
        "TCP/IP protocols, the invention of the World Wide Web by Tim "
        "Berners-Lee in 1989, and the creation of web browsers made the "
        "internet accessible to the general public. By the mid-1990s, "
        "commercial use of the internet exploded, giving rise to email, "
        "online shopping, and search engines. Today, the internet connects "
        "billions of people worldwide, serving as a critical infrastructure "
        "for communication, business, education, and entertainment."
    ),
    "nutrition_basics.txt": (
        "A balanced diet is essential for maintaining good health and "
        "providing the body with the nutrients it needs to function "
        "properly. It typically includes a variety of foods from different "
        "groups, such as fruits, vegetables, whole grains, proteins, and "
        "healthy fats, each providing essential vitamins, minerals, and "
        "energy. Consuming excessive amounts of processed foods, sugar, and "
        "saturated fats can increase the risk of obesity, heart disease, "
        "and other chronic conditions. Nutrition experts recommend eating a "
        "colorful variety of plant-based foods, staying hydrated, and "
        "moderating portion sizes to support long-term health and prevent "
        "diet-related illnesses."
    ),
    "space_exploration.txt": (
        "Space exploration has advanced significantly since the launch of "
        "Sputnik in 1957, marking the beginning of the space race between "
        "the United States and the Soviet Union. Major milestones include "
        "the Apollo moon landings in 1969, the development of reusable "
        "spacecraft, and the establishment of the International Space "
        "Station as a hub for scientific research in orbit. In recent "
        "years, private companies have joined government agencies in "
        "pushing the boundaries of space travel, aiming for missions to "
        "Mars and beyond. These efforts continue to expand our "
        "understanding of the universe while driving innovation in "
        "technology, materials science, and engineering."
    ),
    "cybersecurity_basics.txt": (
        "Cybersecurity involves protecting computer systems, networks, and "
        "data from unauthorized access, theft, or damage. As more aspects "
        "of daily life move online, threats such as malware, phishing "
        "attacks, and data breaches have become increasingly common and "
        "sophisticated. Organizations implement various defense measures, "
        "including firewalls, encryption, multi-factor authentication, and "
        "regular security audits, to safeguard sensitive information. "
        "Individual users are also encouraged to practice good "
        "cybersecurity hygiene, such as using strong passwords and being "
        "cautious with suspicious emails or links. As cyber threats "
        "continue to evolve, the field of cybersecurity remains critical "
        "for protecting privacy, financial systems, and national "
        "infrastructure worldwide."
    ),
}


def build_repository():
    os.makedirs(REPO_DIR, exist_ok=True)
    for filename, content in DOCUMENTS.items():
        filepath = os.path.join(REPO_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"Repository built successfully: {len(DOCUMENTS)} documents written to '{REPO_DIR}'")


if __name__ == "__main__":
    build_repository()
