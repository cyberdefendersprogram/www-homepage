---
layout: blog
title: Cyber Explorers Program
permalink: /cyberexplorers/
---
<br/>
<h1 class="title">Santa Teresa Highschool Program </h1>

Cyber Explorers is a 7-week immersive program aimed at introducing Cyber Security discipline to high school and middle school students.The program doesn’t assume any exposure to computers and computer science. The first 3 sessions are tabletop fun games and exercises introducing concepts of Cyber Security like Cryptography, Network Security, Defense in Depth, Incident Detection, Incident Response et al.

The students will start early in the program on a team project which could be a cybersecurity-related puzzle, poster or a project. This is a deliverable which students are expected to showcase in the Cyber Explorers finale - which would be attended by parents, teachers and potentially rest of the school.

In the latter half of the program after a session on Ethics and signing of a displayable ethics pledge, 
we introduce hands-on exercises and threats which students will replicate in a lab environment. We start the labs with a gentle introduction to shell and operating systems and the drop into network security, log analysis and forensics. 

<br/>
<section>
<div class="container">
    <div class="columns is-multiline is-mobile is-centered">
        <div class="column is-half">
            <figure class="image is-4by3">
            <img src="{{site.url}}{{site.baseurl}}assets/images/santa-teresa-demo-day.jpg"/>
            </figure>
        </div>
        <div class="column is-half">
        <p class="has-text-left">   
            <div>
                <span class="tag is-primary">Its a wrap!</span> The <a href='/cyberexplorers'>Cyber Explorers Program</a> Demo day was on 
                Thursday May 23, from 5-7pm at Santa Teresa High School - Multi Purpose Room, 6150 Snell Ave, San Jose, CA 95123. Its a wrap!
                <br/> <br/>
                <span class="tag is-danger">Photos</span> Program Photos (please add any photos you have): <a href="https://photos.app.goo.gl/LpYBk5bo4a2qXUkB6">Demo Day</a>, 
                    <a href="https://photos.app.goo.gl/5H7TdWcDPStWU7XH9">Escape Room</a>, 
                    <a href="https://photos.app.goo.gl/DMFyErqnozxqpeQs6">Board Games</a>, <a href="https://photos.app.goo.gl/tSnR2eUSRNWH49c26">General</a>.
                <br/> <br/>
                <span class="tag is-warning">Industry Feedback</span>: Check industry evaluations of the projects below.
                <br/> <br/>
                <span class="tag is-secondary">Student Feedback & Opportunities</span>: Links coming soon. We are offering two paid certifications for Comp TIA if the students complete them before July 15 with the help of test out curricula.
                <br/> <br/>
                <span class="tag is-danger">Next Up</span> We are have our annual Hackathon on June 1 at <a href='http://www.merritt.edu' target="_blank">Merritt College</a>. 
                    Application details are available <a href="/hackathons">now</a>! Last day to sign up is <b>Thursday, May 3 , 2019</b>.
            </div>
            </p>
        </div>
    </div>
</div>
</section>

<br/>
<h1 class="title">Student Projects & Industry Feedback</h1>
<div class="columns is-multiline is-mobile is-centered">
{% for project in site.data.cyberexplorers-projects %}  
    <div class="column is-one-third">
        <div class="card is-shady">
            <div class="card-content">
                <div class="content">
                    <p class="is-project-title">{{project.name}}</p>
                    <h6>{{project.members}}</h6>
                    <p>{{project.description}}</p>
                    {% if project.prototype %}
                        <p>Style: <a href='{{project.prototype}}'>{{project.presentation}}</a></p>
                    {% else %}
                        <p>Style: {{project.presentation}}</p>
                    {% endif %}
                    <p class="is-feedback-title">Industry Feeback</p>
                        <ul class="is-feedback-list">
                            {% for point in project.points %}
                            <li>{{point}}</li>
                            {% endfor %}
                        </ul>
                </div>
            </div>
        </div>
    </div>
{% endfor %}
</div>

<h1 class="title">Program details</h1>
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

<h1 class="title">Board Games Introduced</h1>
1. *Security Cards*: This game gives players an opportunity to explore cybersecurity threats from multiple perspectives. The cards have four categories: human impact, adversary's motivation, adversary's resources, adversary's methods. Within these categories, the cards give example scenarios and ideas to jumpstart group games and discussion.
2. *Dx03D*: A tabletop turn-based security game focused on informally introducing students to cybersecurity concepts. The flow of the game requires players to act as attackers at some points, compromising and looting the other team’s infrastructure, and as defenders other times, attempting to patch exploitations and secure against further attacks.
3. *Cards Against Security*: This game is a remix of the popular Cards Against Humanity card game centered around security concepts. The adlib style cards have been reworked to introduce security terms and concepts.
4. *Stratego*: Stratego is a turn-based strategy board game where opposing armies attempt to secure the other team’s flag. This game has a defensive and offensive component where you must strategize where to place your pieces and where to attack your opponent.
5. *Hacker*: In Hacker, players work together to solve progressively harder puzzles. These puzzles follow the flow of creating a program, attack incident and subsequent incident response. As the puzzles get more difficult other obstacles are added forcing players to continuously think outside of the box.
<br/>

<h1 class="title">Project Ideas & Process</h1>
1. Enhance the Cyber Explorers escape room
2. Develop a poster against Cyber Bullying & Teen education.
3. Develop a poster and tools for person cyber security - tools, image guide etc.
4. Work on the Cyber Security games - Security & Privacy cards, Dx0D
5. Ideas from [Students](http://bit.ly/explorerideas2019)
<br/>


