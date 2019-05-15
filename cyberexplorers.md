---
layout: blog
title: Cyber Explorers Program
permalink: /cyberexplorers/
---
<h1 class="title"> Cyber Explorers Santa Teresa Highschool Program </h1>

Cyber Explorers is a 7-week immersive program aimed at introducing Cyber Security discipline to high school and middle school students.The program doesn’t assume any exposure to computers and computer science. The first 3 sessions are tabletop fun games and exercises introducing concepts of Cyber Security like Cryptography, Network Security, Defense in Depth, Incident Detection, Incident Response et al.

The students will start early in the program on a team project which could be a cybersecurity-related puzzle, poster or a project. This is a deliverable which students are expected to showcase in the Cyber Explorers finale - which would be attended by parents, teachers and potentially rest of the school.

In the latter half of the program after a session on Ethics and signing of a displayable ethics pledge, 
we introduce hands-on exercises and threats which students will replicate in a lab environment. We start the labs with a gentle introduction to shell and operating systems and the drop into network security, log analysis and forensics. 

<br/>
<h6 class="subtitle">Program details</h6>
<table class="table is-bordered is-striped">
    <thead>
        <td>Session</td><td>Description</td><td>Notes</td>
    </thead>
    <tbody>
    {% for session in site.data.cyber-explorers-curriculum-program %} 
    <tr>
        <td>{{session.session}}</td>
        <td>{{session.desc | markdownify}}</td>
        <td>{{session.notes | markdownify}}</td>
    </tr>
    {% endfor %}
    </tbody>
</table>

Through the Cyber Explorers, we aim at having a strong industry partnership using guest lectures, hackathons 
and exposing students to career paths. One of the session labs will be tailored as a Cyber Patriots test - which will introduce students to the Cyber Patriots program and other relevant national initiatives. 

The program finale will showcase student work and projects to a broad audience. During the duration of the program, we will offer an extensive cybersecurity library and a raspberry pi based project lab environment. We strive to align this program to other industry efforts and also offer a  hackathon.

<br/>
<h1 class="title">Student Resources </h1>
1. TestOut [Lab Sim](https://cdn.testout.com/client-v5-1-10-555/startlabsim.html)
2. Cyber Defenders [Youtube](https://www.youtube.com/playlist?list=PLyrH1sX4xNPYTxuS3MmowLS8hGlRpAL4R)
3. Session #5 Links
    1. Session #5  Install [PuTTY](https://www.putty.org/)
    2. OverTheWire [Bandit](http://overthewire.org/wargames/bandit/bandit0.html) Challenge
    3. Session #5 Unix Command Challenge - [Challenge Form](https://forms.gle/mdNsK3RvqnmzD7yU7)
    4. Session #5 Optional Challenge - [Challenge Directory ZIP](http://bit.ly/cmd-challenge)
4. Session #6 Links
    1. Session #6 Networking [Challenge Form](http://bit.ly/explorersession6)
<hr/>

<h1 class="title">Project Ideas </h1>
1. Enhance the Cyber Explorers escape room
2. Develop a poster against Cyber Bullying & Teen education.
3. Develop a poster and tools for person cyber security - tools, image guide etc.
4. Work on the Cyber Security games - Security & Privacy cards, Dx0D
5. Ideas from [Students](http://bit.ly/explorerideas2019)
<hr/>

<section>
<div class="container">
<h1 class="title">Projects</h1>
<div class="columns is-multiline is-mobile is-centered">
{% for project in site.data.cyberexplorers-projects %}  
    <div class="column is-one-third">
        <div class="card is-shady">
            <div class="card-content">
                <div class="content">
                    <h4>{{project.name}}</h4>
                    <h6>{{project.members}}</h6>
                    <p>{{project.description}}</p>
                    {% if project.prototype %}
                        <p>Style: <a href='{{project.prototype}}'>{{project.presentation}}</a>
                    {% else %}
                        <p>Style: {{project.presentation}}
                    {% endif %}
                    <br/>Status: <b>{{project.status}}</b></p>
                </div>
            </div>
        </div>
    </div>
{% endfor %}
</div>
</div>
</section>