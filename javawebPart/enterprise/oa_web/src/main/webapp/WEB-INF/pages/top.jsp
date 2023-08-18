<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>中医问答系统</title>

    <link rel="stylesheet" type="text/css" href="/assets/skin/default_skin/css/theme.css">
    <link rel="stylesheet" type="text/css" href="/assets/admin-tools/admin-forms/css/admin-forms.css">
    <link rel="shortcut icon" href="/assets/img/favicon.ico">
    <script type="text/javascript" src="src/main/webapp/vendor/jquery/jquery-1.11.1.min.js"></script>
</head>
<body class="admin-validation-page" data-spy="scroll" data-target="#nav-spy" data-offset="200">
<div id="main">
    <header class="navbar navbar-fixed-top navbar-shadow">
        <div class="navbar-branding">
            <a class="navbar-brand" >
                <b>中医问答系统</b>
            </a>
            <span id="toggle_sidemenu_l" class="ad ad-lines"></span>
        </div>
        <ul class="nav navbar-nav navbar-right">
        </ul>
    </header>
    <aside id="sidebar_left" class="nano nano-light affix">
        <div class="sidebar-left-content nano-content">
            <header class="sidebar-header">
                <div class="sidebar-widget author-widget">
                    <div class="media">
                    </div>
                </div>
                <div class="sidebar-widget search-widget hidden">
                    <div class="input-group">
                        <span class="input-group-addon">
                            <i class="fa fa-search"></i>
                        </span>
                        <input type="text" id="sidebar-search" class="form-control" placeholder="Search...">
                    </div>
                </div>
            </header>
            <ul class="nav sidebar-menu">
                <li class="sidebar-label pt20">图像检测</li>
                <li>
                    <a href="/to_add">
                        <span class="glyphicon glyphicon-book"></span>
                        <span class="sidebar-title">图像分类检测</span>
                        <span class="sidebar-title-tray">
                <span class="label label-xs bg-primary">New</span>
              </span>
                    </a>
                </li>
                <li>
                    <a href="/list.do">
                        <span class="glyphicon glyphicon-book"></span>
                        <span class="sidebar-title">图像分类查看</span>
                        <span class="sidebar-title-tray">
                <span class="label label-xs bg-primary">New</span>
              </span>
                    </a>
                </li>
                <li class="sidebar-label pt15">数据查询</li>
                <li>
                    <a href="/accurate_list.do">
                        <span class="fa fa-calendar"></span>
                        <span class="sidebar-title">准确率查询</span>
                    </a>
                </li>
                <li>
                    <a href="/time_list.do">
                        <span class="fa fa-calendar"></span>
                        <span class="sidebar-title">预测时间查询</span>
                    </a>
                </li>

            </ul>
        </div>
    </aside>
    <section id="content_wrapper">