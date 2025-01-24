---
layout: blog
title: Merritt College Hacker Techniques
permalink: /cis-55/
last_modified_at: 2025-01-23T23:06:20
---
## CIS 55 - Hacker Techniques, Exploits, and Incident Handling.

This is the second course in the infrastructure security major and it will expose students to the major concepts of vulnerability management, exploits, types of malware, and incident response.  Class will use a combination of lectures, required reading, essays, and hands-on labs to teach the course.

Some additional important links below:
- [Merritt College Cybersecurity Path - PDF](/assets/pdf/2024-merritt-career-path.pdf)
- [PG&E $10,000 Scholarship](https://www.pge.com/en/newsroom/press-release-details.b2647d25-741d-4c31-8e5a-100e7291bfaf.html)
- [Microsoft Cybersecurity Grant](https://www.lastmile-ed.org/microsoftcybersecurityscholarship)
- [Collegiate Pentest Competition](https://cp.tc/), [Western Region Collegiate Defense Competition](https://wrccdc.org/), [NCL](https://nationalcyberleague.org/competition)
- Preparation for the competitions [TryHackMe](https://tryhackme.com/), [HackTheBox](https://www.hackthebox.com/)


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
                <span class="tag is-primary">In Progress!</span>
                <br/> <br/>
                <a class="tag is-danger" href="#guest">2024 Final Presentations and Guest Lecture</a>
                <br/> <br/>
                <span class="tag is-danger"><a href="/2023-fall-cis-55">Fall 2023 Class</a></span>
                <span class="tag is-danger"><a href="/2024-spring-cis-55">Spring 2024 Class</a></span>
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
    {% for session in site.data.merritt-cis55-2025-spring-schedule %} 
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
<h1 class="guest">2024 Final Presentations and Guest Lecture</h1>
<iframe width="560" height="315" src="https://www.youtube.com/embed/1yVklgW8JqE?si=i__oFFCj4L6NXSZg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
<hr/>
Last Updated: {{page.last_modified_at}}
