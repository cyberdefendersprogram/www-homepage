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

<div class="toc-container">
  <h2>Guide to Computer Forensics and Investigations</h2>
  <h3>Table of Contents</h3>
  
  <table class="toc-table">
    <thead>
      <tr>
        <th>Chapter</th>
        <th>Title</th>
        <th>Link</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>Introduction to Computer Forensics</td>
        <td><a href="https://docs.google.com/presentation/d/1pHNIbzF5cEToaksguOGkI9wc_aCHQqx4m7lAcijlVsE/edit?usp=sharing" target="_blank">View Chapter</a></td>
      </tr>
      <tr>
        <td>2</td>
        <td>Understanding Computer Investigations</td>
        <td><a href="https://docs.google.com/presentation/d/1ip9kCVCB9bKthdyDQo4cZIYLn9jeCyK5YvReX2tVwN8/edit?usp=sharing" target="_blank">View Chapter</a></td>
      </tr>
      <tr>
        <td>3</td>
        <td>Data Acquisition</td>
        <td><a href="https://docs.google.com/presentation/d/1klS5SLrPRKsAT5K1REQDtQDMJzN-OEM4bgel-Jl9Sug/edit?usp=sharing" target="_blank">View Chapter</a></td>
      </tr>
      <tr>
        <td>4</td>
        <td>Processing Crime and Incident Scenes</td>
        <td><a href="https://docs.google.com/presentation/d/1klS5SLrPRKsAT5K1REQDtQDMJzN-OEM4bgel-Jl9Sug/edit?usp=sharing" target="_blank">View Chapter</a></td>
      </tr>
      <tr>
        <td>5</td>
        <td>Working with Windows and CLI Systems</td>
        <td><a href="https://docs.google.com/presentation/d/13Oq8KTaZ7aiUL9VkqxYUwM7TyhUXa9uDfAXD_Rd-moY/edit?usp=drive_link" target="_blank">View Chapter</a></td>
      </tr>
      <tr>
        <td>6</td>
        <td>Current Computer Forensics Tools</td>
        <td><a href="https://docs.google.com/presentation/d/17XTDJp-bei_W1MGZuppzWHqQPFiPUCaphXjh2KN_ebY/edit?usp=drive_link" target="_blank">View Chapter</a></td>
      </tr>
      <tr>
        <td>7</td>
        <td>Digital Evidence Controls</td>
        <td><a href="https://docs.google.com/presentation/d/1PMd1v3Re6VBjiRxLwpM6ZU7UvYB8BjmQ8yrT3rEqlyM/edit?usp=drive_link" target="_blank">View Chapter</a></td>
      </tr>
      <tr>
        <td>8</td>
        <td>Crime and Incident Scene Processing</td>
        <td><a href="https://docs.google.com/presentation/d/1t3KVl30GJiG1BuX7RBV8yUDc5m53Z-1lcyKpJ4LCHx4/edit?usp=drive_link" target="_blank">View Chapter</a></td>
      </tr>
      <tr>
        <td>9</td>
        <td>Digital Evidence Analysis</td>
        <td><a href="https://docs.google.com/presentation/d/1iRYKpdkOAskVAdb2_mAJE-QfBH4h2ymDUN1tZlGlOuU/edit?usp=drive_link" target="_blank">View Chapter</a></td>
      </tr>
      <tr>
        <td>10</td>
        <td>Virtual Machine Forensics</td>
        <td><a href="https://docs.google.com/presentation/d/1kxMtqvMGZUSuLXKlsXOExhu3Mj-TnpvqxwS4PnfSeOM/edit?usp=drive_link" target="_blank">View Chapter</a></td>
      </tr>
      <tr>
        <td>11</td>
        <td>Network Forensics</td>
        <td><a href="https://docs.google.com/presentation/d/15hLfDw2VRKraJ_BSJjVg7Kmofd36qbzD_MQt1RKkxjg/edit?usp=sharing" target="_blank">View Chapter</a></td>
      </tr>
      <tr>
        <td>12</td>
        <td>Mobile Device Forensics</td>
        <td><a href="https://docs.google.com/presentation/d/1tdbgX0F7SGYN8XyIFhy61p9ebyCdhde5ZPT4dKh-9B8/edit?usp=sharing" target="_blank">View Chapter</a></td>
      </tr>
      <tr>
        <td>13</td>
        <td>Cloud Forensics</td>
        <td><a href="https://docs.google.com/presentation/d/1TXNlHKvSkVnV9VqOG9K8TzRU6Z_YsMeZeEhvWjPrwdE/edit?usp=sharing" target="_blank">View Chapter</a></td>
      </tr>
      <tr>
        <td>14</td>
        <td>Report Writing for High-Tech Investigations</td>
        <td><a href="https://docs.google.com/presentation/d/13502f2-imNIXSpy9K88NUO7pIjETPQ3pzi4b1KRFe_4/edit?usp=sharing" target="_blank">View Chapter</a></td>
      </tr>
      <tr>
        <td>15</td>
        <td>Expert Testimony in High-Tech Investigations</td>
        <td><a href="https://docs.google.com/presentation/d/1bJEykAD34kcuhyACne7PdCTjDNTd9BhO4x-PEMUQjec/edit?usp=sharing" target="_blank">View Chapter</a></td>
      </tr>
      <tr>
        <td>16</td>
        <td>Ethics for High-Technology Investigations</td>
        <td><a href="https://docs.google.com/presentation/d/1Gw6MFyrshF_iBpO9Vf-0zqGnvyFu1FvOuh9rtU2qxtU/edit?usp=sharing" target="_blank">View Chapter</a></td>
      </tr>
    </tbody>
  </table>
</div>

<h1 class="guest">Final Presentations and Guest Lecture</h1>
<hr/>
Last Updated: {{page.last_modified_at}}
