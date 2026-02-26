My In scope - what i'm building
This is my SCOPE (MVP - smallest version of my work, but still looks like a proper system. These will be the only things that im building, which will sow that my FYP is full functional for what I was trying to achieve)

From my list;
1. login with their respective roles - clinicians/managers/admin
2. clinicians dashboard that shows the risk and alert
3. the patients details along with SHAP explanations
4. The Admin is able to change the threshold
5. the Manager is able to view the different metrics

My Out of scope - what im not building

Ive decided im not going to be building a few things, even though its quite good to have inside hospitals, however due to time constraints and the ability I have, I wont be able to 
1. proper security (passowrds/hashing)
2. a full database or audit working system (i'll only do this if time allows)
3. Making my UI absolutely brilliant
4. Real hospital integration like HL7 or FHIR (this is essentialy ways to communicate data like lab results etc. HL7 is more of an older version, so FHIR is more new-er and is more API-friendly too. And the reason why im putting it i the out of scope is because it's 1. alot of work, and 2. it's not really needed to show that my project functions


Now for my demo flow- this is the order of how i'll be narrating things in my video
1. login as the clinician - this will enable them to see the list of patients taht are most at risk from sepsis
2. click on a patient - have a look at the risk, then alerts, and then SHAP reasons, then logout
3. login as Admin - can change the thresholds, and show the alerts changed - logout
4. login as Manager - show the metrics and confusion matrix, logout






References;
https://www.reddit.com/r/projectmanagement/comments/rf3ip3/when_it_comes_to_creating_the_project_scope/
https://rhapsody.health/blog/fhir-vs-hl7-explained/
