---
layout: blog
title: Python Academy
permalink: /cd-growthsector-python-academy/
orientation-preso:
final-preso: 
final-music-embed: |
    <iframe width="100%" height="300" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/1056362899&color=%23ff5500&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></iframe>
last_modified_at: 2022-12-28T02:11:01
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
                <span class="tag is-primary">Coming soon!</span> The <a href=''>Cyber Defenders Python Academy</a>
                <br/> <br/>
                <span class="tag is-danger">Enrollment Complete</span>
                <br/> <br/>
                <span class="tag is-danger">Next Up - First Session</span>
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

Through the Python Academy, we aim at having a comprehensive introduction to Python prorgamming languagee, and exposing students to real-life programming projects. The program finale will showcases student work and projects to a broad audience.
<br/>


<h1 class="title">Project Ideas & Process</h1>

<br/>


<hr/>
Last Updated: {{page.last_modified_at}}