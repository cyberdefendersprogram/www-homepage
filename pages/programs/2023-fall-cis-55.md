---
layout: blog
title: Merritt College Hacker Techniques
permalink: /cis-55/
last_modified_at: 2023-03-24T23:06:20
---
<br/>
<h1 class="title">Merritt College Hacker Techniques </h1>

Hacker Techniques, Exploits, and Incident Handling CIS 55.

This is the second course in the infrastructure security major and it will expose students to the major concepts of vulnerability management, exploits, types of malware, and incident response.  Class will use a combination of lectures, required reading, essays, and hands-on labs to teach the course.

<br/>
<section>
<div class="container">
    <div class="columns is-multiline is-mobile is-centered">
        <div class="column is-half">
            <figure class="image">
            <img src="{{site.url}}{{site.baseurl}}assets/images/merritt-cis-55.jpeg"/>
            </figure>
        </div>
        <div class="column is-half">
        <p class="has-text-left">   
            <div>
                <span class="tag is-primary">Completed!</span>
                <br/> <br/>
                <a class="tag is-danger" href="#December+15">Final Presentations and Guest Lecture</a>
                <br/> <br/>
                <span class="tag is-danger"><a href="/2023-spring-cis-55">Spring 2023 Class</a></span>
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
    {% for session in site.data.merritt-cis55-2023-fall-schedule %} 
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

<br/>


<hr/>
Last Updated: {{page.last_modified_at}}