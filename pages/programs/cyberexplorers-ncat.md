---
layout: blog
title: Cyber Explorers NCAT Program
permalink: /cyberexplorers-ncat/
final-preso: https://docs.google.com/presentation/d/1R_uyEyVU5Ywl1-WVpL0c6SvuhB8mHsJDBn4zkl-9qBg/edit?usp=sharing
final-music-embed: |
    <iframe width="100%" height="300" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/1056362899&color=%23ff5500&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></iframe>
cert-students:
    - Tony Riddick
    - Keone Best
    - Mahmoud Amadou
    - Danya Grady
    - Emani
    - Isaiah Daniels
    - Mekhi Roberts
    - Corey Scott
    - Iyona Kane
    - Caidyn Anderson
    - Jahquinn Tookes
    - Ibrahim Abdul Mutakabbir
    - Ryan Travers
    - Tariq
    - Elijah Booker
    - Christina Smith- Chrissy
    - DeMichael Morgan
    - D’Shani Smith
    - Caidyn Anderson
    - Mitchell
    - Daiyana Brooks
    - Tandeka Boko (Auditor)
last_modified_at: 2023-05-12T21:54:08
---
# 2023 CyberAggies Summer Bridge Program
{: .title .has-text-danger .mt-4 .is-capitalized}

Cyber Explorers NCAT is a 4-week summer program aimed at introducing Cyber Security discipline to  students at [NCAT](https://www.ncat.edu/). We call this cohort the **CYBER AGGIES**. The program doesn’t assume any exposure to computers and computer science. The first few sessions are tabletop fun games and exercises introducing concepts of Cyber Security like Cryptography, Network Security, Defense in Depth, Incident Detection, Incident Response et al. We will also intorduction to python programming in this class as a student request.

The students will start early in the program on a project which could be a cybersecurity-related puzzle, poster or a project. This is a deliverable which students are expected to showcase in the Cyber Explorers finale.

Here are details from [2022 program]({{site.url}}{{site.baseurl}}/cyberexplorers-ncat-2022/).

## Dates
{: .list-disc}
 -  Week 1: Tuesday 6/20 - Thursday 6/22 (3 days)
 -  Week 2: Mon  6/26 - Thurs 6/29 (4 days)
 -  Week 3: Wed 7/05 - Thurs 7/07 (2 days)
 -  Week 4: Mon 7/10 - Thurs 7/13 (4 days)
 -  Week 5: Mon 7/17 - Thurs 7/20 (4 days)

*Class timing is 8-9am PST.*

<br/>
<section>
<div class="container">
    <div class="columns is-mobile is-centered">
        <div class="column">
            <figure class="image is-128x128">
                <img src="{{site.url}}{{site.baseurl}}assets/images/ncat.png"/>
            </figure>
        </div>
        <div class="column">
            <p class="has-text-left">   
                <div>
                    <span class="tag is-primary">In Progress!</span><a href=''>The Cyber Explorers NCAT Program</a>
                    <br/> <br/>
                    <span class="tag is-danger">Enrollment Completed</span>
                    <br/> <br/>
                    <span class="tag is-danger">Next Up Session #3</span>
                </div>
            </p>
        </div>
    </div>
</div>
</section>

<br/>
<h1 class="title">Program details</h1>
<table class="table is-bordered is-striped">
    <thead>
        <td>Session</td><td>Description</td><td>Notes from Sessions</td>
    </thead>
    <tbody>
    {% for session in site.data.cyber-explorers-ncat %} 
    <tr>
        <td><a id="{{session.session| url_encode}}" href="#{{session.session | url_encode}}">{{session.session}}</a></td>
        <td>{{session.desc | markdownify}}</td>
        <td>{{session.notes | markdownify}}</td>
    </tr>
    {% endfor %}
    </tbody>
</table>

Through the Cyber Explorers, we aim at having a comprehensive introduction to cyber security, related hackathons and exposing students to cyber security career paths. The program finale will showcases student work and projects to a broad audience.
<br/>


<h1 class="title">Project Ideas & Process</h1>
1. Enhance the Cyber Explorers escape room
2. Develop a poster against Cyber Bullying & Teen education.
3. Develop a poster and tools for person cyber security - tools, image guide etc.
4. Work on the Cyber Security games - Security & Privacy cards, Dx0D
5. Past Ideas from [Students](http://bit.ly/explorerideas2019)
<br/>

<h1 class="title"><a id="projects" href="#projects">Student Projects</a></h1>
<p>Use this form to submit <a href="">reviews</a></p>
<a class="tag is-info" href="">ON YOUTUBE - LIVE PRESENTATION</a>
<table class="table is-bordered is-striped">
    <thead>
        <td>No</td><td>Title</td><td>Presentation</td><td>Team</td>
    </thead>
    <tbody>
    {% assign sorted_projects = site.data.2023-cyberaggies-projects  | sort: "order" %}
    {% for project in sorted_projects %} 
        {% if project.order != "" %}
      <tr>
        <td><a id="{{project.title| url_encode}}" href="#{{project.title | url_encode}}">{{project.order}}</a></td>
        <td>{{project.title}}</td>
        <td>{% if project.presentation %}
            <a href="{{project.presentation}}">Link</a>
            {% else %}
            Presentation
            {% endif %}
        </td>
        <td>{{project.team | markdownify}}</td>
    </tr>
        {% endif %}
    {% endfor %}
    </tbody>
</table>
<br/>
<hr/>

<h1 class="title"><a id="progress" href="#certificates">Student Progress</a></h1>
<table class="table is-bordered is-striped">
    <thead>
        <td>Student</td><td>Progress</td>
    </thead>
    <tbody>
    {% for student in page.cert-students %} 
    <tr>
        <td><a id="{{student | url_encode}}" href="#{{student | url_encode}}">{{student}}</a></td>
        <td><a href="{{site.url}}{{site.baseurl}}assets/images/gs-certs/png/{{student | replace: ' ','_'}}.png">Progress Link</a></td>
    </tr>
    {% endfor %}
    </tbody>
</table>
<br/>
<hr/>
Last Updated: {{page.last_modified_at}}