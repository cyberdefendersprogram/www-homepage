---
layout: blog
title: Merritt College Hacker Techniques
permalink: /cis-52/
last_modified_at: 2023-08-11 08:12:05+00:00
---
<br/>
<h1 class="title">CIS 52 Cloud Security (Merritt College)</h1>

Cloud Security (Fall 2023)

This is the second course in the infrastructure security major and it will expose students to the major concepts of Cloud Security.  Class will use a combination of lectures, required reading, essays, and hands-on labs to teach the course.

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
                <span class="tag is-danger">Enrolling!</span>
                <br/> <br/>
                <a class="tag is-danger">Guest Lecture</a>
                <br/> <br/>
                <span class="tag is-danger">Next Up Winter 2023</span>
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
    {% for session in site.data.merritt-cis52-schedule %} 
    <tr>
        <td><a id="{{session.session| url_encode}}" href="#{{session.session | url_encode}}">{{session.session}}</a></td>
        <td>{{session.desc | markdownify}}</td>
        <td>{{session.notes | markdownify}}</td>
    </tr>
    {% endfor %}
    </tbody>
</table>

<br/>


<hr/>
Last Updated: {{page.last_modified_at}}