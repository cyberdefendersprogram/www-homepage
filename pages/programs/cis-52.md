---
layout: blog
title: Merritt College CIS 52
permalink: /cis-52/
last_modified_at: 2023-08-11 08:12:05+00:00
---
<br/>
<h1 class="title">CIS 52 Cloud Security (Merritt College)</h1>

Cloud Security (Fall 2025)

This is the second course in the infrastructure security major and it will expose students to the major concepts of Cloud Security.  Class will use a combination of lectures, required reading, essays, and hands-on labs to teach the course.

Some additional important links below:
- [Fall 2024 CIS 52 Class](/2024-fall-cis-52) [Fall 2023 CIS 52 Class](/2023-fall-cis-52)
- [Merritt College Cybersecurity Path - PDF](/assets/pdf/2024-merritt-career-path.pdf)
- [Microsoft Cybersecurity Grant](https://www.lastmile-ed.org/microsoftcybersecurityscholarship)
- [Collegiate Pentest Competition](https://cp.tc/), [Western Region Collegiate Defense Competition](https://wrccdc.org/), [NCL](https://nationalcyberleague.org/competition)
- Preparation for the competitions [TryHackMe](https://tryhackme.com/), [HackTheBox](https://www.hackthebox.com/)

<br/>
<section>
<div class="container">
    <div class="columns is-multiline is-mobile is-centered">
        <div class="column is-half">
            <figure class="image">
            <img src="{{site.url}}{{site.baseurl}}assets/images/merritt-cis-52.jpg"/>
            </figure>
        </div>
        <div class="column is-half">
        <p class="has-text-left">   
            <div>
                <span class="tag is-danger">In Session!</span>
                <br/> <br/>
                <a class="tag is-info" href="#guestlecture">Guest Lecture</a>
                <br/> <br/>
                <a class="tag is-danger" href="/cis-52-quiz" target="_blank">Quiz</a>
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
    {% for session in site.data.merritt-cis52-schedule %} 
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

<h2 id="guestlecture" class="subtitle">Course Outline</h2>
<p> On Sep 19, 2025 we will have guest lecture from Keith Hodo.</p

<hr/>
Last Updated: {{page.last_modified_at}}
