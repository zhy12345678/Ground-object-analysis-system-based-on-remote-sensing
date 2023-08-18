<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>

<jsp:include page="top.jsp"/>
<section id="content" class="table-layout animated fadeIn">
    <div class="tray tray-center">
        <div class="content-header">
            <h2> 准确率图形化展示 </h2>
            <p class="lead"></p>
        </div>
        <div class="admin-form theme-primary mw1000 center-block" style="padding-bottom: 175px;">
            <div class="panel  heading-border">
                <div class="panel-menu">
                    <div class="row">
                    </div>
                </div>
                <div class="panel-body pn">
                    <table id="message-table" class="table admin-form theme-warning tc-checkbox-1">
                        <thead>
                        <tr class="">
                            <th class="hidden-xs">图片名称</th>
                            <th class="hidden-xs">模型名称</th>
                            <th class="hidden-xs">准确率展示</th>
                        </tr>
                        </thead>
                        <tbody>
                        <c:forEach items="${list}" var="picture">
                            <tr class="message-unread">
                                <th class="text-center fw600">${picture.filename}</th>
                                <th class="text-center fw600">${picture.modelname}</th>
                                <th ><img id="images" alt="" src="/upload/${picture.accurate }" height="300px" width="600px"></th>
                            </tr>
                        </c:forEach>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</section>

<jsp:include page="bottom.jsp"/>