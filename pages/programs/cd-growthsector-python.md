---
layout: blog
title: Python Academy
permalink: /cd-growthsector-python-academy/
orientation-preso:
final-preso: 
final-music-embed:
last_modified_at: 2023-01-13T21:13:38
---
<br/>
<h1 class="title">Python Academy</h1>

Python is a widely used and fast growing programming language which is in high demand for jobs.  Many of the STEM Core internships utilize Python programming to solve problems ranging from industrial automation to artificial intelligence.  In this 2-week Python Academy, STEM Core students will learn Python in a live, synchronous course from a technology professional and cybersecurity consultant.  The course is free to participants and will result in handson Python proficiency and a certificate displayable on sites like LinkedIn.  This interactive course will consist of both lecture sessions and supported active programming time to solve fun problems!

<br/>
<section>
<div class="container">
    <div class="columns is-multiline is-mobile is-centered">
        <div class="column is-half">
            <figure class="image is-square">
            <img src="{{site.url}}{{site.baseurl}}assets/images/gs-python.png"/>
            </figure>
        </div>
        <div class="column is-half">
        <p class="has-text-left">   
            <div>
                <span class="tag is-primary">In Progress!</span> The <a href=''>Cyber Defenders Python Academy</a>
                <br/> <br/>
                <span class="tag is-danger">Enrollment Complete</span>
                <br/> <br/>
                <span class="tag is-danger">Next Up - Sessions and Office Hours</span>
                <br/> <br/>
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
        <td>Session</td><td>Description</td><td>Notes</td>
    </thead>
    <tbody>
    {% for session in site.data.cyber-defenders-python-academy %} 
    <tr>
        <td><a id="{{session.session| url_encode}}" href="#{{session.session | url_encode}}">{{session.session}}</a></td>
        <td>{{session.desc | markdownify}}</td>
        <td>{{session.notes | markdownify}}</td>
    </tr>
    {% endfor %}
    </tbody>
</table>

<br/>
Through the Python Academy, we aim at having a comprehensive introduction to Python prorgamming languagee, and exposing students to real-life programming projects. The program finale will showcases student work and projects to a broad audience.
<br/>


<h1 class="title">Project Ideas & Process</h1>
<table class="table is-bordered is-striped">
    <thead>
        <td>Title</td><td>Description</td><td>Team</td>
    </thead>
    <tbody>
    {% for project in site.data.cyber-defenders-python-academy-projects %} 
    <tr>
        <td><a id="{{project.title| url_encode}}" href="#{{project.title | url_encode}}">{{project.title}}</a></td>
        <td>{{project.desc | markdownify}}</td>
        <td>{{project.team | markdownify}}</td>
    </tr>
    {% endfor %}
    </tbody>
</table>
<br/>
<hr/>
Last Updated: {{page.last_modified_at}}