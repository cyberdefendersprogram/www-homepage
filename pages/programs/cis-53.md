---
layout: blog
title: Merritt College CIS 52
permalink: /cis-52/
last_modified_at: 2024-10-22T06:20:36Z 
---
<br/>
<h1 class="title">CIS 53 Intrusion Detection in Depth (Merritt College)</h1>

Intrusion Detection (Fall 2024)
CIS 53 - Intrusion Detection In-Depth is a comprehensive course focused on the detection and analysis of network-based threats. Students will gain hands-on experience in network traffic analysis, intrusion detection systems (IDS/IPS), and advanced monitoring tools such as Wireshark, tcpdump, Snort, and Zeek. The course covers key topics like threat intelligence, indicators of compromise (IoCs), and effective security architecture for proactive monitoring. With guest speakers and practical labs, students will develop critical skills to detect zero-day threats and enhance network security. The course culminates in a hands-on intrusion detection challenge.

Some additional important links below:
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
            <img src="{{site.url}}{{site.baseurl}}assets/images/merritt-cis-53.jpg"/>
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


<hr/>
Last Updated: {{page.last_modified_at}}
