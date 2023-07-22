---
layout: blog
title: Python Academy
permalink: /cd-growthsector-python-academy/
cert-students:
    - TBA
last_modified_at: 2023-03-24T22:28:35
---
<br/>
<h1 class="title">Python Academy</h1>

Python is a widely used and fast growing programming language which is in high demand for jobs.  Many of the STEM Core internships utilize Python programming to solve problems ranging from industrial automation to artificial intelligence.  In this 2-week Python Academy, STEM Core students will learn Python in a live, synchronous course from a technology professional and cybersecurity consultant.  The course is free to participants and will result in handson Python proficiency and a certificate displayable on sites like LinkedIn.  This interactive course will consist of both lecture sessions and supported active programming time to solve fun problems!

<br/>
<section>
<div class="container">
    <div class="columns is-multiline is-mobile is-centered">
        <div class="column is-one-third">
            <figure class="image">
            <img src="{{site.url}}{{site.baseurl}}assets/images/gs-python.png"/>
            </figure>
        </div>
        <div class="column is-two-third">
        <p class="has-text-left">   
            <div>
                <span class="tag is-danger">Check 2022 Program!</span> <a href='/2022-cd-growthsector-python-academy/'>2022 Cyber Defenders Python Academy</a>
                <br/> <br/>
                <span class="tag is-danger">Enrollment Open!!</span>
                <br/> <br/>
                <span class="tag is-danger">Next Up - Info Session!</span>
                <br/> <br/>
            </div>
            </p>
        </div>
    </div>
</div>
</section>

<br/>
<h1 class="title">Program details</h1>
<div>
    <p class="tag is-info">Class from 9-11am PST weekdays and Drop-in tutor session 1:30-2:30pm PST</p>
    <ul>
        <li>Each student will work on a hands-on program</li>
        <li>All lectures will be recorded</li>
        <li>Each student receives a certificate on completion alteast 80% of assignments and final project</li>
    </ul>
</div>
<br/>
<table class="table is-bordered is-striped">
    <thead>
        <td>Session</td><td>Description</td><td>Notes from Session</td>
    </thead>
    <tbody>
    {% for session in site.data.gs-python-schedule %} 
    <tr>
        <td><a id="{{session.session| url_encode}}" href="#{{session.session | url_encode}}">{{session.session}}</a></td>
        <td>{{session.desc | markdownify}}</td>
        <td>Agenda: {{session.notes | markdownify}}
            <ol>
            {%if session.slides_link != "" %}
                <li><a href="{{session.slides_link}}" class="tag is-info">Slides</a></li>
                <li><a href="{{session.next1_link}}" class="tag is-warning">{{session.next1_name}}</a></li>                
                <li><a href="{{session.recording}}" class="tag is-info">Recording</a></li>
            {%endif%}
            </ol>
        </td>
    </tr>
    {% endfor %}
    </tbody>
</table>

<br/>
Through the Python Academy, we aim at having a comprehensive introduction to Python prorgamming languagee, and exposing students to real-life programming projects. The program finale will showcases student work and projects to a broad audience.
<br/>

<h1 class="title"><a id="projects" href="#projects">2022 Student Projects</a></h1>
<p>Use this form to submit <a href="https://docs.google.com/forms/d/e/1FAIpQLSeA_lMQWVrwFkEvh9cOjFISOkabSacRtEo0Tuk_7-AIlfp-Zg/viewform">reviews</a></p>
<a class="tag is-danger" href="https://us02web.zoom.us/rec/share/GSHhYNjl_vOBhD73XP7vZlmfzYooYYHv-J3Pt7oeS6FuJ_xAMq9Av4TM7pz4Wq2e.y5pWWIbKhp2zndGE">ZOOM RECORDED PRESENTATIONS</a>
<a class="tag is-info" href="https://www.youtube.com/watch?v=8L_1c8vztM4">ON YOUTUBE - RECORDED PRESENTATIONS</a>
<table class="table is-bordered is-striped">
    <thead>
        <td>Title</td><td>Code</td><td>Description</td><td>Team</td>
    </thead>
    <tbody>
    {% for project in site.data.cyber-defenders-python-academy-projects %} 
    <tr>
        <td><a id="{{project.title| url_encode}}" href="#{{project.title | url_encode}}">{{project.title}}</a></td>
        <td>{% if project.presentation_url %}
            <a href="{{project.presentation_url}}">Code</a>
            {% else %}
            Presentation
            {% endif %}
            </td>
        <td>{{project.desc | markdownify}}</td>
        <td>{{project.team | markdownify}}</td>
    </tr>
    {% endfor %}
    </tbody>
</table>
<br/>
<hr/>

<h1 class="title"><a id="certificates" href="#certificates">Student Certificates</a></h1>
<table class="table is-bordered is-striped">
    <thead>
        <td>Student</td><td>Certificate</td>
    </thead>
    <tbody>
    {% for student in page.cert-students %} 
    <tr>
        <td><a id="{{student | url_encode}}" href="#{{student | url_encode}}">{{student}}</a></td>
        <td><a href="{{site.url}}{{site.baseurl}}assets/images/gs-certs/png/{{student | replace: ' ','_'}}.png">certificate</a></td>
    </tr>
    {% endfor %}
    </tbody>
</table>
<br/>

<hr/>
Last Updated: {{page.last_modified_at}}