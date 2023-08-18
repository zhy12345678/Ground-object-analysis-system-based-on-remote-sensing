<%@ taglib prefix="form" uri="http://www.springframework.org/tags/form" %>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<jsp:include page="top.jsp"/>
<script type="text/javascript">
    //选择图片，马上预览
    var image = '';
    function upload(file) {
        if(!file.files || !file.files[0]){
            return;
        }
        var reader = new FileReader();
       //读取文件过程方法
        reader.onload = function (e) {
            document.getElementById('Img').src = e.target.result;
            image = e.target.result;
        }
        reader.readAsDataURL(file.files[0])
    }


    function modalAddPicture() {
        console.log('调试区');
        var formdata = new FormData();
        formdata.append("file",$("#fileName1")[0].files[0]);
        formdata.append("modelname",$("#modelname option:selected").val());
        formdata.append("datamarket",$("#datamarket option:selected").val());
        console.log(formdata);
        // return;
        $.ajax({
            async:false,
            cache:false,
            url:'addProduct.do',
            data:formdata,
            type:'POST',
            contentType: false,
            processData: false,
            success:function (result) {
                if(result.code == 100){
                    alert("成功")
                }else {
                    alert("成功")
                }
            }
        });
    }
</script>
<section id="content" class="table-layout animated fadeIn">
    <div class="tray tray-center">
        <div class="content-header">
            <h2> 图片分类检测</h2>
            <p class="lead"></p>
        </div>
        <div class="admin-form theme-primary mw1000 center-block" style="padding-bottom: 175px;">
            <div class="panel heading-border">
                <form:form  modelAttribute="picture"  id="admin-form" name="addForm" method="post" enctype="multipart/form-data" >
                    <div class="panel-body bg-light">
                        <div class="section-divider mt20 mb40">
                            <span> 基本信息选择 </span>
                        </div>
                        <div class="section row">
                            <div class="col-md-6">
                                <label for="filename" class="field prepend-icon">
                                    <form:input id="fileName1" name="file" type="file" path="filename" cssClass="gui-input" placeholder="选择需要预测文件" onchange="upload(this)" accept="image/*" />
                                    <label for="filename" class="field-icon">
                                        <i class="fa fa-user"></i>
                                    </label>
                                </label>
                            </div>
                            <div>
                                <img id="Img" width="180px" height="180px"/>
                            </div>
                        </div>
                        <div class="section row">
                            <div class="col-md-6">
                                <label for="modelname" class="field select">
                                    <form:select path="modelname" items="${mlist}" id="modelname" name="modelname"  cssClass="gui-input" placeholder="选择模型"/>
                                    <i class="arrow double"></i>
                                </label>
                            </div>
                        </div>
                        <div class="section row">
                            <div class="col-md-6">
                                <label for="datamarket" class="field select">
                                    <form:select path="datamarket" items="${dlist}" id="datamarket" name="datamarket" cssClass="gui-input" placeholder="选择数据集"/>
                                    <i class="arrow double"></i>
                                </label>
                            </div>
                        </div>
                        <div class="panel-footer text-right">
                            <input class="button" type="button" onclick="modalAddPicture()" value="预测">
                        </div>
                    </div>
                </form:form>
            </div>
        </div>
    </div>
</section>

<jsp:include page="bottom.jsp"/>
