---
layout: blog
title: Merritt College Computer Forensics Fundamentals
permalink: /cis-60/
last_modified_at: 2024-04-02T21:55:43Z
---
<br/>
<h1 class="title">Computer Forensics Fundamentals </h1>

Welcome to CIS 60 Computer Forensics Fundamentals! During this course, you will be presented with an overview of computer forensics, including computer investigation processes, operating systems boot processes, and disk structures. You'll learn data acquisition tools, techniques. You'll develop your skills of analysis &technical writing. We will discuss the legal and ethical obligations of a forensic analyst. And, finally, the objectives of the International Association of Computer Investigative Specialists (IACIS) certification.

<br/>
<section>
<div class="container">
    <div class="columns is-multiline is-mobile is-centered">
        <div class="column is-half">
            <figure class="image">
            <img src="{{site.url}}{{site.baseurl}}assets/images/merritt-cis-60.jpeg"/>
            </figure>
        </div>
        <div class="column is-half">
        <p class="has-text-left">   
            <div>
                <span class="tag is-primary">In Progress!</span>
                <br/> <br/>
                <a class="tag is-info" href="/2024S-cis-60/">Spring 2024 Class</a>
                <br/><br/>
                <a class="tag is-danger" href="#guest">Final Presentations and Guest Lecture</a>
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
        <td>Session</td><td>Description</td><td>Notes</td><td>Slides</td><td>Recording</td>
    </thead>
    <tbody>
    {% for session in site.data.merritt-cis60-2025-spring-schedule %} 
    <tr>
        <td><a id="{{session.session| url_encode}}" href="#{{session.session | url_encode}}">{{session.session}}</a></td>
        <td>{{session.desc | markdownify}}</td>
        <td>{{session.notes | markdownify}}</td>
        {%if session.slides_link != "" %}
        <td><a href="{{session.slides_link}}" class="tag is-info">Slides</a></td>
        {% endif %}
        {%if session.recording_link != "" %}
        <td><a href="{{session.recording_link}}" class="tag is-info">Recording</a></td>
        {% endif %}
    </tr>
    {% endfor %}
    </tbody>
</table>
<hr/>
<h1 class="guest">Final Presentations and Guest Lecture</h1>

<hr/>
Last Updated: {{page.last_modified_at}}
